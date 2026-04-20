#!/bin/bash
# Batch 1 (8 GPUs): UViT + PixArt experiments
#   0: uvit-god-ddim150          (DDIM steps ablation)
#   1: uvit-god-ddim250          (DDIM steps baseline)
#   2: uvit-god-crop03           (crop ratio ablation)
#   3: pixart-god-ddim150        (DDIM steps ablation)
#   4: pixart-god-ddim250        (DDIM steps baseline)
#   5: pixart-god-crop03         (crop ratio ablation)
#   6: uvit-bold5000             (cross-model benchmark)
#   7: pixart-bold5000           (cross-model benchmark)

cd /scratch/georgeli/mind-vis

LOG_DIR="logs/ablation_batch1"
mkdir -p "$LOG_DIR"

echo "Launching 8 experiments. Logs -> $LOG_DIR"

WANDB_NAME="uvit-god-ddim150" CUDA_VISIBLE_DEVICES=0 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone uvit --ddim_steps 150 \
    > "$LOG_DIR/uvit-god-ddim150.log" 2>&1 &

WANDB_NAME="uvit-god-ddim250" CUDA_VISIBLE_DEVICES=1 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone uvit --ddim_steps 250 \
    > "$LOG_DIR/uvit-god-ddim250.log" 2>&1 &

WANDB_NAME="uvit-god-crop03" CUDA_VISIBLE_DEVICES=2 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone uvit --crop_ratio 0.3 \
    > "$LOG_DIR/uvit-god-crop03.log" 2>&1 &

WANDB_NAME="pixart-god-ddim150" CUDA_VISIBLE_DEVICES=3 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone pixart-adaln-single --ddim_steps 150 \
    > "$LOG_DIR/pixart-god-ddim150.log" 2>&1 &

WANDB_NAME="pixart-god-ddim250" CUDA_VISIBLE_DEVICES=4 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone pixart-adaln-single --ddim_steps 250 \
    > "$LOG_DIR/pixart-god-ddim250.log" 2>&1 &

WANDB_NAME="pixart-god-crop03" CUDA_VISIBLE_DEVICES=5 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone pixart-adaln-single --crop_ratio 0.3 \
    > "$LOG_DIR/pixart-god-crop03.log" 2>&1 &

WANDB_NAME="uvit-bold5000" CUDA_VISIBLE_DEVICES=6 \
    python code/stageB_ldm_finetune.py --dataset BOLD5000 --backbone uvit --batch_size 25 \
    > "$LOG_DIR/uvit-bold5000.log" 2>&1 &

WANDB_NAME="pixart-bold5000" CUDA_VISIBLE_DEVICES=7 \
    python code/stageB_ldm_finetune.py --dataset BOLD5000 --backbone pixart-adaln-single --batch_size 25 \
    > "$LOG_DIR/pixart-bold5000.log" 2>&1 &

echo "All 8 experiments launched."
jobs -l
wait
echo "All batch 1 experiments complete."
