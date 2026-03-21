#!/bin/bash
#SBATCH --job-name=nmiai-finetune-dino
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail

# Fine-tunes DINOv2 on shelf product crops with supervised classification head.
# Exports backbone FP16 + classifier head weights.
#
# Usage:
#   sbatch scripts/train_dino.slurm
#   sbatch --export=ALL,UNFREEZE_BLOCKS=2,EPOCHS=20,BATCH_SIZE=256 scripts/train_dino.slurm

MODEL=${MODEL:-vit_base_patch14_dinov2}
UNFREEZE_BLOCKS=${UNFREEZE_BLOCKS:-2}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-256}
IMG_SIZE=${IMG_SIZE:-518}
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-1}
LR_BACKBONE=${LR_BACKBONE:-2e-5}
LR_HEAD=${LR_HEAD:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
DROPOUT=${DROPOUT:-0.1}
MIN_CROP_SIZE=${MIN_CROP_SIZE:-20}
WORKERS=${WORKERS:-4}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config: model=${MODEL}, unfreeze_blocks=${UNFREEZE_BLOCKS}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, img_size=${IMG_SIZE}"
echo "LR: backbone=${LR_BACKBONE}, head=${LR_HEAD}  label_smoothing=${LABEL_SMOOTHING}  dropout=${DROPOUT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs:"
    nvidia-smi -L
fi

GRAD_CHECKPOINT_FLAG=""
[[ "${GRAD_CHECKPOINT}" == "1" ]] && GRAD_CHECKPOINT_FLAG="--grad-checkpoint"

if command -v uv >/dev/null 2>&1; then
    uv run python train/train_dino.py \
        --model "${MODEL}" \
        --unfreeze-blocks "${UNFREEZE_BLOCKS}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --img-size "${IMG_SIZE}" \
        --lr-backbone "${LR_BACKBONE}" \
        --lr-head "${LR_HEAD}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --label-smoothing "${LABEL_SMOOTHING}" \
        --dropout "${DROPOUT}" \
        --min-crop-size "${MIN_CROP_SIZE}" \
        --workers "${WORKERS}" \
        ${GRAD_CHECKPOINT_FLAG}
else
    python train/train_dino.py \
        --model "${MODEL}" \
        --unfreeze-blocks "${UNFREEZE_BLOCKS}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --img-size "${IMG_SIZE}" \
        --lr-backbone "${LR_BACKBONE}" \
        --lr-head "${LR_HEAD}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --label-smoothing "${LABEL_SMOOTHING}" \
        --dropout "${DROPOUT}" \
        --min-crop-size "${MIN_CROP_SIZE}" \
        --workers "${WORKERS}" \
        ${GRAD_CHECKPOINT_FLAG}
fi
