#!/bin/bash
#SBATCH --job-name=nmiai-irl-embed
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail

# Optional overrides at submit time:
#   sbatch slurm_scripts/build_irl_embeddings.slurm
#   sbatch --export=ALL,MODEL=vit_small_patch14_dinov2 slurm_scripts/build_irl_embeddings.slurm
#   sbatch --export=ALL,MIN_CROP=32 slurm_scripts/build_irl_embeddings.slurm

MODEL=${MODEL:-vit_base_patch14_dinov2}
IMG_SIZE=${IMG_SIZE:-518}
BATCH=${BATCH:-64}
MIN_CROP=${MIN_CROP:-20}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Building IRL embeddings: model=${MODEL}, img_size=${IMG_SIZE}, batch=${BATCH}, min_crop=${MIN_CROP}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs from nvidia-smi:"
    nvidia-smi -L
fi

if command -v uv >/dev/null 2>&1; then
    uv run python src/build_irl_embeddings.py \
        --model "${MODEL}" \
        --img-size "${IMG_SIZE}" \
        --batch-size "${BATCH}" \
        --min-crop-size "${MIN_CROP}"
else
    python src/build_irl_embeddings.py \
        --model "${MODEL}" \
        --img-size "${IMG_SIZE}" \
        --batch-size "${BATCH}" \
        --min-crop-size "${MIN_CROP}"
fi
