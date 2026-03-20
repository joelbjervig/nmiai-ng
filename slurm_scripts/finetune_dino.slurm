#!/bin/bash
#SBATCH --job-name=nmiai-finetune-dino
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail

# Fine-tunes DINOv2 on shelf product crops using ArcFace loss.
# Overwrites model/<MODEL>_fp16.pth with the fine-tuned FP16 backbone.
# After this, re-run build_irl_embeddings.slurm to rebuild reference embeddings.
#
# Usage:
#   sbatch slurm_scripts/finetune_dino.slurm
#   sbatch --export=ALL,UNFREEZE_BLOCKS=6,EPOCHS=80 slurm_scripts/finetune_dino.slurm
#
# Chain with build_irl_embeddings afterwards:
#   sbatch --dependency=afterok:<job_id> slurm_scripts/build_irl_embeddings.slurm

MODEL=${MODEL:-vit_base_patch14_dinov2}
UNFREEZE_BLOCKS=${UNFREEZE_BLOCKS:-4}
EPOCHS=${EPOCHS:-60}
BATCH_SIZE=${BATCH_SIZE:-64}
IMG_SIZE=${IMG_SIZE:-336}
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-1}
LR_BACKBONE=${LR_BACKBONE:-2e-5}
LR_HEAD=${LR_HEAD:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
ARCFACE_S=${ARCFACE_S:-30.0}
ARCFACE_M=${ARCFACE_M:-0.4}
MIN_CROP_SIZE=${MIN_CROP_SIZE:-20}
WORKERS=${WORKERS:-4}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config: model=${MODEL}, unfreeze_blocks=${UNFREEZE_BLOCKS}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, img_size=${IMG_SIZE}"
echo "LR: backbone=${LR_BACKBONE}, head=${LR_HEAD}  ArcFace: s=${ARCFACE_S}, m=${ARCFACE_M}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs:"
    nvidia-smi -L
fi

GRAD_CHECKPOINT_FLAG=""
[[ "${GRAD_CHECKPOINT}" == "1" ]] && GRAD_CHECKPOINT_FLAG="--grad-checkpoint"

if command -v uv >/dev/null 2>&1; then
    uv run python src/finetune_dino.py \
        --model "${MODEL}" \
        --unfreeze-blocks "${UNFREEZE_BLOCKS}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --img-size "${IMG_SIZE}" \
        --lr-backbone "${LR_BACKBONE}" \
        --lr-head "${LR_HEAD}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --arcface-s "${ARCFACE_S}" \
        --arcface-m "${ARCFACE_M}" \
        --min-crop-size "${MIN_CROP_SIZE}" \
        --workers "${WORKERS}" \
        ${GRAD_CHECKPOINT_FLAG}
else
    python src/finetune_dino.py \
        --model "${MODEL}" \
        --unfreeze-blocks "${UNFREEZE_BLOCKS}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --img-size "${IMG_SIZE}" \
        --lr-backbone "${LR_BACKBONE}" \
        --lr-head "${LR_HEAD}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --arcface-s "${ARCFACE_S}" \
        --arcface-m "${ARCFACE_M}" \
        --min-crop-size "${MIN_CROP_SIZE}" \
        --workers "${WORKERS}" \
        ${GRAD_CHECKPOINT_FLAG}
fi
