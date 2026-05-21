#!/usr/bin/env bash
# SHAH-Sudoku experiment suite on a single A100.
#
# Phase 1: tune λ for the collision rule loss at N_TUNE (eval_acc-based selection).
# Phase 2: tune λ for the permanent rule loss at N_TUNE.
# Phase 3: final training at N_FINAL for three conditions:
#            (A) vanilla MDM
#            (B) collision loss @ λ*_collision
#            (C) permanent loss @ λ*_permanent
#          Each final run predicts on the SHAH test set at the end.
#
# Preconditions (NOT done by this script):
#   $DATASET_DIR/sudoku_train.csv   ≥ N_FINAL SHAH puzzles (recommended: 100k)
#   $DATASET_DIR/sudoku_test.csv    SHAH test puzzles      (recommended: 10k)
#   e.g.:
#     python3 scripts/convert_shah_sudoku.py $DATASET_DIR $DATASET_DIR \
#         --train-samples 100000 --test-samples 10000 --seed 42
#
# Env vars (all optional):
#   SEED              (default: 42)
#   GPU               (default: 3)
#   N_TUNE            tune-phase train size            (default: 10000)
#   N_FINAL           final-training train size        (default: 100000)
#   MAX_STEPS_TUNE    optimizer steps per tune run     (default: 5000)
#   MAX_STEPS_FINAL   optimizer steps per final run    (default: 27600)
#   LAMBDAS_COLL      λ grid for collision loss        (default: "0.01 0.1 1.0 10.0")
#   LAMBDAS_PERM      λ grid for permanent loss        (default: "0.001 0.01 0.1 1.0")
#   DATASET_DIR       data folder                      (default: $HOME/datasets/data/)
#   RUNS_DIR          log/ckpt root                    (default: $HOME/logs)
#   SUITE_TAG         override suite dir name
set -euo pipefail

export WANDB_DISABLED=true

SEED="${SEED:-42}"
GPU="${GPU:-3}"
N_TUNE="${N_TUNE:-10000}"
N_FINAL="${N_FINAL:-100000}"
MAX_STEPS_TUNE="${MAX_STEPS_TUNE:-5000}"
MAX_STEPS_FINAL="${MAX_STEPS_FINAL:-27600}"
LAMBDAS_COLL_STR="${LAMBDAS_COLL:-0.01 0.1 1.0 10.0}"
LAMBDAS_PERM_STR="${LAMBDAS_PERM:-0.001 0.01 0.1 1.0}"
RUNS_DIR="${RUNS_DIR:-$HOME/logs}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/data/}"

SUITE_TAG="${SUITE_TAG:-sudoku-shah-suite-s${SEED}-$(date +%Y%m%d-%H%M%S)}"
suite_dir="$RUNS_DIR/$SUITE_TAG"
mkdir -p "$suite_dir"
suite_log="$suite_dir/suite.log"
summary_tsv="$suite_dir/summary.tsv"
lambdas_json="$suite_dir/lambdas.json"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

{
    echo "==== SHAH suite config ===="
    echo "suite_dir         = $suite_dir"
    echo "SEED              = $SEED"
    echo "GPU               = $GPU"
    echo "N_TUNE / N_FINAL  = $N_TUNE / $N_FINAL"
    echo "MAX_STEPS_TUNE    = $MAX_STEPS_TUNE"
    echo "MAX_STEPS_FINAL   = $MAX_STEPS_FINAL"
    echo "LAMBDAS_COLL      = $LAMBDAS_COLL_STR"
    echo "LAMBDAS_PERM      = $LAMBDAS_PERM_STR"
    echo "DATASET_DIR       = $DATASET_DIR"
    echo "commit            = $(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "============================"
} | tee "$suite_log"

# ---- Precondition: SHAH CSVs present and large enough ----
for f in sudoku_train.csv sudoku_test.csv; do
    if [[ ! -f "$DATASET_DIR/$f" ]]; then
        echo "ERROR: $DATASET_DIR/$f missing. Run scripts/convert_shah_sudoku.py first." | tee -a "$suite_log"
        exit 1
    fi
done
train_rows=$(($(wc -l < "$DATASET_DIR/sudoku_train.csv") - 1))
test_rows=$(($(wc -l < "$DATASET_DIR/sudoku_test.csv") - 1))
echo "train rows = $train_rows, test rows = $test_rows" | tee -a "$suite_log"
if (( train_rows < N_FINAL )); then
    echo "ERROR: train CSV has only $train_rows rows, need >= $N_FINAL." | tee -a "$suite_log"
    exit 1
fi

# ---- Helper: pick best λ from a tune-phase directory ----
pick_best_lambda() {
    local tune_dir="$1"
    python3 - "$tune_dir" <<'PY'
import json, os, sys, glob
tune_dir = sys.argv[1]
best_lam, best_acc = None, -1.0
rows = []
for d in sorted(glob.glob(os.path.join(tune_dir, "lam*"))):
    lam = os.path.basename(d).replace("lam", "")
    state_path = os.path.join(d, "trainer_state.json")
    if not os.path.exists(state_path):
        rows.append((lam, None, "no trainer_state.json"))
        continue
    try:
        h = json.load(open(state_path))["log_history"]
        evals = [e["eval_acc"] for e in h if "eval_acc" in e]
        acc = max(evals) if evals else None
    except Exception as e:
        rows.append((lam, None, f"err:{e}"))
        continue
    rows.append((lam, acc, ""))
    if acc is not None and acc > best_acc:
        best_acc, best_lam = acc, lam
for lam, acc, note in rows:
    print(f"  λ={lam}\tbest_eval_acc={acc}\t{note}", file=sys.stderr)
if best_lam is None:
    sys.exit("no valid eval_acc found in any tune run")
print(best_lam)
PY
}

# ---- Helper: run a single training (delegates to train-mdm-a100.sh) ----
# args: run_dir run_tag max_samples max_steps [rule_loss_kind] [rule_loss_weight] [skip_predict]
run_train() {
    local run_dir="$1" run_tag="$2" max_samples="$3" max_steps="$4"
    local rl_kind="${5:-}" rl_weight="${6:-}" skip_predict="${7:-0}"
    mkdir -p "$run_dir"
    local env_args=(
        RUN_DIR="$run_dir"
        RUN_TAG="$run_tag"
        SEED="$SEED"
        GPU="$GPU"
        MAX_STEPS="$max_steps"
        MAX_SAMPLES="$max_samples"
        DATASET_DIR="$DATASET_DIR"
        SKIP_PREDICT="$skip_predict"
    )
    if [[ -n "$rl_kind" ]]; then
        env_args+=(RULE_LOSS_KIND="$rl_kind")
    fi
    if [[ -n "$rl_weight" ]]; then
        env_args+=(RULE_LOSS_WEIGHT="$rl_weight")
    fi
    env "${env_args[@]}" bash scripts/sudoku/train-mdm-a100.sh
}

# ---- Helper: tune one loss kind over its λ grid ----
# Writes selected λ to "$suite_dir/tune-${kind}/best_lambda.txt".
# Does NOT use command substitution — so train output streams to terminal/log
# normally and errors are visible.
tune_loss() {
    local kind="$1" lambdas_str="$2"
    local tune_dir="$suite_dir/tune-${kind}"
    mkdir -p "$tune_dir"
    echo "==== $(date -Is)  Phase 1/2: tuning λ for ${kind} loss ====" | tee -a "$suite_log"
    for lam in $lambdas_str; do
        local run_dir="$tune_dir/lam${lam}"
        echo "---- tune ${kind} λ=${lam} ----" | tee -a "$suite_log"
        if ! run_train "$run_dir" "tune-${kind}-lam${lam}" "$N_TUNE" "$MAX_STEPS_TUNE" "$kind" "$lam" 1; then
            echo "FAILED at tune-${kind}-lam${lam} — see $run_dir/train.log" | tee -a "$suite_log"
            exit 1
        fi
    done
    echo "==== picking λ* for ${kind} ====" | tee -a "$suite_log"
    pick_best_lambda "$tune_dir" > "$tune_dir/best_lambda.txt" 2>>"$suite_log"
    local best_lam
    best_lam="$(cat "$tune_dir/best_lambda.txt")"
    echo "λ*_${kind} = ${best_lam}" | tee -a "$suite_log"
}

# ---- Phase 1: collision λ tune ----
tune_loss collision "$LAMBDAS_COLL_STR"
lam_coll="$(cat "$suite_dir/tune-collision/best_lambda.txt")"

# ---- Phase 2: permanent λ tune ----
tune_loss permanent "$LAMBDAS_PERM_STR"
lam_perm="$(cat "$suite_dir/tune-permanent/best_lambda.txt")"

# ---- Persist selected λs ----
python3 - <<PY > "$lambdas_json"
import json
print(json.dumps({"collision": float("$lam_coll"), "permanent": float("$lam_perm")}, indent=2))
PY
echo "selected lambdas written to $lambdas_json:" | tee -a "$suite_log"
cat "$lambdas_json" | tee -a "$suite_log"

# ---- Phase 3: final training (3 runs) ----
echo "==== $(date -Is)  Phase 3: final training (N=$N_FINAL, steps=$MAX_STEPS_FINAL) ====" | tee -a "$suite_log"

final_vanilla="$suite_dir/final-vanilla"
final_collision="$suite_dir/final-collision-lam${lam_coll}"
final_permanent="$suite_dir/final-permanent-lam${lam_perm}"

echo "---- final: vanilla MDM ----" | tee -a "$suite_log"
run_train "$final_vanilla"   "final-vanilla"                  "$N_FINAL" "$MAX_STEPS_FINAL" ""          ""         0

echo "---- final: collision @ λ=${lam_coll} ----" | tee -a "$suite_log"
run_train "$final_collision" "final-collision-lam${lam_coll}" "$N_FINAL" "$MAX_STEPS_FINAL" collision  "$lam_coll" 0

echo "---- final: permanent @ λ=${lam_perm} ----" | tee -a "$suite_log"
run_train "$final_permanent" "final-permanent-lam${lam_perm}" "$N_FINAL" "$MAX_STEPS_FINAL" permanent  "$lam_perm" 0

# ---- Summary ----
echo "==== $(date -Is)  collecting summary ====" | tee -a "$suite_log"
python3 - <<PY | tee -a "$suite_log" | tee "$summary_tsv"
import json, os
suite="$suite_dir"
runs=[("vanilla","$final_vanilla"),
      ("collision-lam${lam_coll}","$final_collision"),
      ("permanent-lam${lam_perm}","$final_permanent")]
print("condition\tpredict_acc\tpredict_path")
for name, d in runs:
    p = os.path.join(d, "sudoku_test", "predict_results.json")
    try:
        r = json.load(open(p))
        acc = r.get("predict_acc", r.get("predict_accuracy", "?"))
    except Exception as e:
        acc = f"err:{e}"
    print(f"{name}\t{acc}\t{p}")
PY

echo "suite complete: $suite_dir" | tee -a "$suite_log"
