#!/usr/bin/env bash
# Data-efficiency sweep on single A100, compute-matched.
# 5 training-set sizes × same MAX_STEPS. Works for baseline and rule-loss conditions.
#
# Env vars (all optional):
#   SEED                subsample + HF seed        (default: 42)
#   GPU                 CUDA device index          (default: 3)
#   MAX_STEPS           cap optimizer steps        (default: 27600 = Nfull @ 300 epochs)
#   RUNS_DIR            log/ckpt root              (default: $HOME/logs)
#   DATASET_DIR         data folder                (default: $HOME/datasets/data/)
#   RULE_LOSS_WEIGHT    if set & > 0 → rule-loss run; else baseline
#   SWEEP_TAG           override default name for the sweep dir
set -euo pipefail

export WANDB_DISABLED=true

SEED="${SEED:-42}"
GPU="${GPU:-3}"
MAX_STEPS="${MAX_STEPS:-27600}"
RUNS_DIR="${RUNS_DIR:-$HOME/logs}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/data/}"

# Condition: baseline or ruleloss-lam<λ>
if [[ -n "${RULE_LOSS_WEIGHT:-}" && "${RULE_LOSS_WEIGHT}" != "0" && "${RULE_LOSS_WEIGHT}" != "0.0" ]]; then
    condition="ruleloss-lam${RULE_LOSS_WEIGHT}"
else
    condition="baseline"
    unset RULE_LOSS_WEIGHT
fi

SWEEP_TAG="${SWEEP_TAG:-${condition}-s${SEED}-steps${MAX_STEPS}}"
sweep_name="sudoku-mdm-sweep-${SWEEP_TAG}-$(date +%Y%m%d-%H%M%S)"
sweep_dir="$RUNS_DIR/$sweep_name"
mkdir -p "$sweep_dir"

sweep_log="$sweep_dir/sweep.log"
echo "sweep_dir        = $sweep_dir" | tee -a "$sweep_log"
echo "condition        = $condition" | tee -a "$sweep_log"
echo "MAX_STEPS        = $MAX_STEPS" | tee -a "$sweep_log"
echo "RULE_LOSS_WEIGHT = ${RULE_LOSS_WEIGHT:-0.0}" | tee -a "$sweep_log"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

# Fixed 5-point sweep: {1k, 3k, 10k, 30k, full=100k}. Smallest first so the pipeline is
# de-risked on a cheap run before committing to the full one.
runs=(
    "1000    N1k"
    "3000    N3k"
    "10000   N10k"
    "30000   N30k"
    ""       #full (Nfull)
)

for spec in "${runs[@]}"; do
    read -r n tag <<< "$spec"
    if [[ -z "$n" ]]; then
        unset MAX_SAMPLES
        tag="Nfull"
    else
        export MAX_SAMPLES="$n"
    fi

    run_dir="$sweep_dir/$tag"
    mkdir -p "$run_dir"

    echo "==== $(date -Is)  starting $tag  (MAX_SAMPLES=${MAX_SAMPLES:-full}, MAX_STEPS=$MAX_STEPS) ====" | tee -a "$sweep_log"

    # Forward rule_loss_weight only if set (keeps baseline code path pristine when off)
    extra_env=()
    if [[ -n "${RULE_LOSS_WEIGHT:-}" ]]; then
        extra_env+=(RULE_LOSS_WEIGHT="$RULE_LOSS_WEIGHT")
    fi

    env \
        RUN_DIR="$run_dir" \
        RUN_TAG="$tag" \
        SEED="$SEED" \
        GPU="$GPU" \
        MAX_STEPS="$MAX_STEPS" \
        DATASET_DIR="$DATASET_DIR" \
        "${extra_env[@]}" \
        bash scripts/sudoku/train-mdm-a100.sh \
            || { echo "FAILED at $tag (exit $?)" | tee -a "$sweep_log"; exit 1; }

    acc=$(python3 -c "
import json
try:
    d = json.load(open('$run_dir/sudoku_test/predict_results.json'))
    print(d.get('predict_acc', d.get('predict_accuracy', '?')))
except Exception as e:
    print(f'parse_err:{e}')
")
    line="$(date -Is)  done  $tag  N=${MAX_SAMPLES:-full}  steps=$MAX_STEPS  cond=$condition  test_acc=$acc"
    echo "$line" | tee -a "$sweep_log"
    echo "$line" >> "$sweep_dir/summary.tsv"
done

echo "sweep complete: $sweep_dir" | tee -a "$sweep_log"
