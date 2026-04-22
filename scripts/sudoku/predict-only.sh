#!/usr/bin/env bash
# Run --do_predict against an already-trained checkpoint (no training).
# Use after a training run completed but the predict step failed, to recover
# predict_results.json without redoing training.
#
# Usage:
#   RUN_DIR=/absolute/path/to/existing/run GPU=3 bash scripts/sudoku/predict-only.sh
#
# Required env var:
#   RUN_DIR     the existing run directory (must contain a checkpoint)
# Optional env vars:
#   GPU         CUDA device index          (default: 3)
#   SEED        HF seed                    (default: 42)
#   DATASET_DIR data folder                (default: $HOME/datasets/data/)
set -euo pipefail

export WANDB_DISABLED=true

# Same PATH fix as train-mdm-a100.sh
diff_env_bin="${CONDA_DIFFUSION_BIN:-$HOME/miniconda3/envs/diffusion/bin}"
if [[ -d "$diff_env_bin" ]]; then
    export PATH="$diff_env_bin:$PATH"
fi
echo "python3 = $(which python3)"

: "${RUN_DIR:?RUN_DIR is required (absolute path to completed training run)}"
GPU="${GPU:-3}"
SEED="${SEED:-42}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/data/}"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "RUN_DIR does not exist: $RUN_DIR" >&2
    exit 1
fi

eval_dir="$RUN_DIR/sudoku_test"
mkdir -p "$eval_dir"

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

CUDA_VISIBLE_DEVICES="$GPU" \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 164 \
    --dataset sudoku_test \
    --dataset_dir "$DATASET_DIR" \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir "$eval_dir" \
    --checkpoint_dir "$RUN_DIR" \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding True \
    --seed "$SEED" \
    2>&1 | tee "$eval_dir/eval.log"

echo "done: $eval_dir"
