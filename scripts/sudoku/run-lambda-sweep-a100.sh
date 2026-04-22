#!/usr/bin/env bash
# λ tuning sweep for the rule-loss term.
# Runs rule-loss training at N=10k for each candidate λ, same seed, same compute budget.
# Pick λ* = value with highest best-seen eval_acc, then use it for the Phase 4 sweep.
#
# Env vars (all optional, pass-through to train-mdm-a100.sh):
#   SEED        (default: 42)
#   GPU         (default: 3)
#   MAX_STEPS   (default: 27600)
#   N_TUNE      train-set size to tune on (default: 10000)
#   LAMBDAS     space-separated list of λ values (default: "0.01 0.1 1.0 10.0")
#   RUNS_DIR    (default: $HOME/logs)
#   DATASET_DIR (default: $HOME/datasets/data/)
set -euo pipefail

export WANDB_DISABLED=true

SEED="${SEED:-42}"
GPU="${GPU:-3}"
MAX_STEPS="${MAX_STEPS:-27600}"
N_TUNE="${N_TUNE:-10000}"
LAMBDAS_STR="${LAMBDAS:-0.01 0.1 1.0 10.0}"
RUNS_DIR="${RUNS_DIR:-$HOME/logs}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/data/}"

sweep_name="sudoku-mdm-lambda-tune-N${N_TUNE}-s${SEED}-$(date +%Y%m%d-%H%M%S)"
sweep_dir="$RUNS_DIR/$sweep_name"
mkdir -p "$sweep_dir"
sweep_log="$sweep_dir/sweep.log"

echo "sweep_dir  = $sweep_dir" | tee -a "$sweep_log"
echo "N_TUNE     = $N_TUNE"   | tee -a "$sweep_log"
echo "MAX_STEPS  = $MAX_STEPS" | tee -a "$sweep_log"
echo "LAMBDAS    = $LAMBDAS_STR" | tee -a "$sweep_log"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

for lam in $LAMBDAS_STR; do
    tag="lam${lam}"
    run_dir="$sweep_dir/$tag"
    mkdir -p "$run_dir"

    echo "==== $(date -Is)  starting $tag ====" | tee -a "$sweep_log"

    env \
        RUN_DIR="$run_dir" \
        RUN_TAG="$tag" \
        SEED="$SEED" \
        GPU="$GPU" \
        MAX_STEPS="$MAX_STEPS" \
        MAX_SAMPLES="$N_TUNE" \
        RULE_LOSS_WEIGHT="$lam" \
        DATASET_DIR="$DATASET_DIR" \
        bash scripts/sudoku/train-mdm-a100.sh \
            || { echo "FAILED at $tag (exit $?)" | tee -a "$sweep_log"; exit 1; }

    # Record final test_acc + best eval_acc during training
    metrics=$(python3 -c "
import json
run='$run_dir'
try:
    pred = json.load(open(run + '/sudoku_test/predict_results.json')).get('predict_acc', '?')
except Exception as e:
    pred = f'err:{e}'
try:
    h = json.load(open(run + '/trainer_state.json'))['log_history']
    evals = [e['eval_acc'] for e in h if 'eval_acc' in e]
    best = max(evals) if evals else '?'
except Exception as e:
    best = f'err:{e}'
print(f'test_acc={pred}  best_eval_acc={best}')
")
    line="$(date -Is)  done  $tag  λ=$lam  $metrics"
    echo "$line" | tee -a "$sweep_log"
    echo "$line" >> "$sweep_dir/summary.tsv"
done

echo "lambda sweep complete: $sweep_dir" | tee -a "$sweep_log"
echo "Pick λ* = the value with highest best_eval_acc (or test_acc if they disagree)." | tee -a "$sweep_log"
