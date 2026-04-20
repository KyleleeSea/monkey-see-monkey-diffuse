#!/bin/bash
# Batch 2 (4 GPUs): DiT experiments
#   0: dit-god-ddim150           (DDIM steps ablation)
#   1: dit-god-ddim250           (DDIM steps baseline)
#   2: dit-god-crop03            (crop ratio ablation)
#   3: dit-bold5000              (cross-model benchmark)

cd /scratch/georgeli/mind-vis

LOG_DIR="logs/ablation_batch2"
mkdir -p "$LOG_DIR"

echo "Launching 4 DiT experiments. Logs -> $LOG_DIR"

WANDB_NAME="dit-god-ddim150" CUDA_VISIBLE_DEVICES=0 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone dit-adaln-zero --ddim_steps 150 \
    > "$LOG_DIR/dit-god-ddim150.log" 2>&1 &

WANDB_NAME="dit-god-ddim250" CUDA_VISIBLE_DEVICES=1 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone dit-adaln-zero --ddim_steps 250 \
    > "$LOG_DIR/dit-god-ddim250.log" 2>&1 &

WANDB_NAME="dit-god-crop03" CUDA_VISIBLE_DEVICES=2 \
    python code/stageB_ldm_finetune.py --dataset GOD --backbone dit-adaln-zero --crop_ratio 0.3 \
    > "$LOG_DIR/dit-god-crop03.log" 2>&1 &

WANDB_NAME="dit-bold5000" CUDA_VISIBLE_DEVICES=3 \
    python code/stageB_ldm_finetune.py --dataset BOLD5000 --backbone dit-adaln-zero --batch_size 25 \
    > "$LOG_DIR/dit-bold5000.log" 2>&1 &

echo "All 4 DiT experiments launched."
jobs -l
wait
echo "All batch 2 experiments complete."
