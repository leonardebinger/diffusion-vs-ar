#!/usr/bin/env bash
# Single-A100 MDM training, paper-matching hyperparameters.
# Global batch size 1024 preserved via gradient accumulation (64 * 16).
# After training, runs --do_predict on sudoku_test.
#
# Env vars (all optional):
#   EPOCHS        training epochs                (default: 300)
#   MAX_SAMPLES   truncate train set to N rows   (default: unset = full)
#   SEED          subsample + HF seed            (default: 42)
#   GPU           CUDA device index              (default: 3)
#   DATASET_DIR   data folder                    (default: $HOME/datasets/data/)
#   RUNS_DIR      log/ckpt root                  (default: $HOME/logs)
#   RUN_DIR      full path for this run's outputs (overrides auto-naming)
#   RUN_TAG       label inserted into run dir    (default: auto: Nfull | N<MAX_SAMPLES>)
set -euo pipefail

export WANDB_DISABLED=true

EPOCHS="${EPOCHS:-300}"
SEED="${SEED:-42}"
GPU="${GPU:-3}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/data/}"
RUNS_DIR="${RUNS_DIR:-$HOME/logs}"

if [[ -n "${MAX_SAMPLES:-}" ]]; then
    default_tag="N${MAX_SAMPLES}"
    max_samples_arg=(--max_samples "${MAX_SAMPLES}")
else
    default_tag="Nfull"
    max_samples_arg=()
fi
RUN_TAG="${RUN_TAG:-$default_tag}"

if [[ -z "${RUN_DIR:-}" ]]; then
    run_name="sudoku-mdm-${RUN_TAG}-s${SEED}-ep${EPOCHS}-$(date +%Y%m%d-%H%M%S)"
    RUN_DIR="$RUNS_DIR/$run_name"
fi
mkdir -p "$RUN_DIR"

{
    echo "==== run config ===="
    echo "RUN_DIR      = $RUN_DIR"
    echo "DATASET_DIR  = $DATASET_DIR"
    echo "EPOCHS       = $EPOCHS"
    echo "MAX_SAMPLES  = ${MAX_SAMPLES:-(full)}"
    echo "SEED         = $SEED"
    echo "GPU          = $GPU"
    echo "commit       = $(git -C "$(dirname "$0")/../.." rev-parse HEAD 2>/dev/null || echo '?')"
    echo "===================="
} | tee "$RUN_DIR/run_config.txt"

# ---- Training ----
CUDA_VISIBLE_DEVICES="$GPU" \
accelerate launch --num_processes 1 --mixed_precision fp16 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_train \
    --dataset sudoku_train \
    --dataset_dir "$DATASET_DIR" \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir "$RUN_DIR" \
    --overwrite_cache \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs "$EPOCHS" \
    --plot_loss \
    --run_name "$(basename "$RUN_DIR")" \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha 0.25 \
    --gamma 1 \
    --seed "$SEED" \
    "${max_samples_arg[@]}" \
    2>&1 | tee "$RUN_DIR/train.log"

# ---- Test-set evaluation ----
eval_dir="$RUN_DIR/sudoku_test"
mkdir -p "$eval_dir"

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

echo "done: $RUN_DIR"
