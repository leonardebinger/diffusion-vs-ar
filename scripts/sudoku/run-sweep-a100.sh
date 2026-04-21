#!/usr/bin/env bash
# Baseline data-efficiency sweep on single A100.
# Runs 4 MDM trainings sequentially: 1k, 10k, 100k, full (unset MAX_SAMPLES).
# All other hyperparameters paper-matching via train-mdm-a100.sh.
set -euo pipefail

export WANDB_DISABLED=true

SEED="${SEED:-42}"
GPU="${GPU:-3}"
EPOCHS="${EPOCHS:-300}"
RUNS_DIR="${RUNS_DIR:-$HOME/logs}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/data/}"

sweep_name="sudoku-mdm-sweep-s${SEED}-ep${EPOCHS}-$(date +%Y%m%d-%H%M%S)"
sweep_dir="$RUNS_DIR/$sweep_name"
mkdir -p "$sweep_dir"

sweep_log="$sweep_dir/sweep.log"
echo "sweep_dir = $sweep_dir" | tee -a "$sweep_log"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

# Probe the full dataset size once, then derive sweep sizes as full/1000, full/100, full/10, full.
n_full=$(DATASET_DIR="$DATASET_DIR" python3 - <<'PY'
import json, os, sys
from datasets import load_dataset
root = os.environ['DATASET_DIR']
info = json.load(open(os.path.join(root, 'dataset_info.json')))
entry = info['sudoku_train']
fname = entry['file_name']
path = os.path.join(root, fname)
ext = fname.rsplit('.', 1)[-1].lower()
type_map = {'json': 'json', 'jsonl': 'json', 'csv': 'csv', 'txt': 'text', 'tsv': 'csv'}
if ext not in type_map:
    print(f"unknown extension: {ext}", file=sys.stderr); sys.exit(1)
ds = load_dataset(type_map[ext], data_files=path, split='train')
print(len(ds))
PY
)
if ! [[ "$n_full" =~ ^[0-9]+$ ]]; then
    echo "failed to determine dataset size; got: $n_full" | tee -a "$sweep_log"
    exit 1
fi
n_div10=$(( n_full / 10 ))
n_div100=$(( n_full / 100 ))
n_div1000=$(( n_full / 1000 ))
echo "dataset size: full=$n_full  /10=$n_div10  /100=$n_div100  /1000=$n_div1000" | tee -a "$sweep_log"

# (MAX_SAMPLES, tag) pairs; empty MAX_SAMPLES = full dataset. Smallest first.
runs=(
    "$n_div1000   Ndiv1000"
    "$n_div100    Ndiv100"
    "$n_div10     Ndiv10"
    ""            #full
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

    echo "==== $(date -Is)  starting $tag  (MAX_SAMPLES=${MAX_SAMPLES:-full}) ====" | tee -a "$sweep_log"

    RUN_DIR="$run_dir" \
    RUN_TAG="$tag" \
    SEED="$SEED" \
    GPU="$GPU" \
    EPOCHS="$EPOCHS" \
    DATASET_DIR="$DATASET_DIR" \
    bash scripts/sudoku/train-mdm-a100.sh \
        || { echo "FAILED at $tag (exit $?)" | tee -a "$sweep_log"; exit 1; }

    # Append a one-line summary: tag, N, final predict_acc (parsed from predict_results.json)
    acc=$(python3 -c "
import json, sys
try:
    d = json.load(open('$run_dir/sudoku_test/predict_results.json'))
    print(d.get('predict_acc', d.get('predict_accuracy', '?')))
except Exception as e:
    print(f'parse_err:{e}')
")
    echo "$(date -Is)  done  $tag  N=${MAX_SAMPLES:-full}  test_acc=$acc" | tee -a "$sweep_log" \
        >> "$sweep_dir/summary.tsv"
done

echo "sweep complete: $sweep_dir" | tee -a "$sweep_log"
