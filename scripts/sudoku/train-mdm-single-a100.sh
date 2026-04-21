#!/usr/bin/env bash
# Single-A100 smoke test: 1 epoch, paper-matching hyperparameters.
# Global batch size 1024 preserved via gradient accumulation (64 * 16).
# After training, runs --do_predict on sudoku_test for a test-accuracy number.
set -euo pipefail

export WANDB_DISABLED=true

DATASET_DIR="${DATASET_DIR:-$HOME/datasets/data/}"
RUNS_DIR="${RUNS_DIR:-$HOME/logs}"
run_name="sudoku-mdm-1ep-$(date +%Y%m%d-%H%M%S)"
run_dir="$RUNS_DIR/$run_name"
mkdir -p "$run_dir"

echo "dataset_dir = $DATASET_DIR"
echo "run_dir     = $run_dir"

# ---- Training ----
CUDA_VISIBLE_DEVICES=0 \
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
    --output_dir "$run_dir" \
    --overwrite_cache \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_steps 200 \
    --learning_rate 1e-3 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --run_name "$run_name" \
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
    2>&1 | tee "$run_dir/train.log"

# ---- Test-set evaluation (--do_predict on sudoku_test) ----
eval_dir="$run_dir/sudoku_test"
mkdir -p "$eval_dir"

CUDA_VISIBLE_DEVICES=0 \
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
    --checkpoint_dir "$run_dir" \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding True \
    2>&1 | tee "$eval_dir/eval.log"
