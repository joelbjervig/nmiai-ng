#!/bin/bash
#SBATCH --job-name=nmiai-build-emb
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail

# Build reference embeddings from training crops using finetuned DINOv2.
#
# Usage:
#   sbatch scripts/build_embeddings.slurm
#   sbatch --export=ALL,WEIGHTS=model/vit_base_patch14_dinov2_fp16.pth scripts/build_embeddings.slurm

WEIGHTS=${WEIGHTS:-model/vit_base_patch14_dinov2_fp16.pth}
BATCH_SIZE=${BATCH_SIZE:-64}
IMG_SIZE=${IMG_SIZE:-518}
LAYER=${LAYER:-11}
MAX_PER_CLASS=${MAX_PER_CLASS:-10}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Backbone: ${WEIGHTS}"

if command -v uv >/dev/null 2>&1; then
    uv run python scripts/build_embeddings.py \
        --weights "${WEIGHTS}" \
        --batch-size "${BATCH_SIZE}" \
        --img-size "${IMG_SIZE}"
else
    python scripts/build_embeddings.py \
        --weights "${WEIGHTS}" \
        --batch-size "${BATCH_SIZE}" \
        --img-size "${IMG_SIZE}"
fi
