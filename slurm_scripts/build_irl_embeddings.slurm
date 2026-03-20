#!/bin/bash
#SBATCH --job-name=nmiai-irl-embed
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
set -euo pipefail

# Optional overrides at submit time:
#   sbatch slurm_scripts/build_irl_embeddings.slurm
#   sbatch --export=ALL,MODEL=vit_small_patch14_dinov2 slurm_scripts/build_irl_embeddings.slurm
#   sbatch --export=ALL,MIN_CROP=32 slurm_scripts/build_irl_embeddings.slurm
#   sbatch --export=ALL,WEIGHTS=model/vit_base_patch14_dinov2_finetune_best.pth slurm_scripts/build_irl_embeddings.slurm

MODEL=${MODEL:-vit_base_patch14_dinov2}
IMG_SIZE=${IMG_SIZE:-518}
BATCH=${BATCH:-64}
MIN_CROP=${MIN_CROP:-20}
MAX_CROPS_PER_CAT=${MAX_CROPS_PER_CAT:-}
WEIGHTS=${WEIGHTS:-}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Building IRL embeddings: model=${MODEL}, weights=${WEIGHTS:-default}, img_size=${IMG_SIZE}, batch=${BATCH}, min_crop=${MIN_CROP}, max_crops_per_cat=${MAX_CROPS_PER_CAT:-all}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs from nvidia-smi:"
    nvidia-smi -L
fi

WEIGHTS_ARG=""
if [[ -n "$WEIGHTS" ]]; then
    WEIGHTS_ARG="--weights ${WEIGHTS}"
fi

MAX_CROPS_ARG=""
if [[ -n "$MAX_CROPS_PER_CAT" ]]; then
    MAX_CROPS_ARG="--max-crops-per-cat ${MAX_CROPS_PER_CAT}"
fi

if command -v uv >/dev/null 2>&1; then
    uv run python src/build_irl_embeddings.py \
        --model "${MODEL}" \
        --img-size "${IMG_SIZE}" \
        --batch-size "${BATCH}" \
        --min-crop-size "${MIN_CROP}" \
        ${WEIGHTS_ARG} \
        ${MAX_CROPS_ARG}
else
    python src/build_irl_embeddings.py \
        --model "${MODEL}" \
        --img-size "${IMG_SIZE}" \
        --batch-size "${BATCH}" \
        --min-crop-size "${MIN_CROP}" \
        ${WEIGHTS_ARG} \
        ${MAX_CROPS_ARG}
fi
