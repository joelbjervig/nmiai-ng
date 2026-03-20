#!/bin/bash
#SBATCH --job-name=nmiai-embed
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail

# Optional overrides at submit time:
#   sbatch build_embeddings.slurm
#   sbatch --export=ALL,MODEL=vit_small_patch14_dinov2,BATCH=64 build_embeddings.slurm
#   sbatch --export=ALL,EXPORT_ONNX=1 build_embeddings.slurm

MODEL=${MODEL:-vit_base_patch14_dinov2}
IMG_SIZE=${IMG_SIZE:-518}
BATCH=${BATCH:-32}
EXPORT_ONNX=${EXPORT_ONNX:-0}

if [[ -z "${DEVICE:-}" ]]; then
    DEVICE=cuda
fi

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Building embeddings: model=${MODEL}, img_size=${IMG_SIZE}, batch=${BATCH}, device=${DEVICE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs from nvidia-smi:"
    nvidia-smi -L
fi

EXTRA_ARGS=""
if [[ "${EXPORT_ONNX}" == "1" ]]; then
    EXTRA_ARGS="--export-onnx"
    echo "ONNX export enabled"
fi

if command -v uv >/dev/null 2>&1; then
    uv run python src/build_embeddings.py \
        --model "${MODEL}" \
        --device "${DEVICE}" \
        --img-size "${IMG_SIZE}" \
        --batch-size "${BATCH}" \
        ${EXTRA_ARGS}
else
    python src/build_embeddings.py \
        --model "${MODEL}" \
        --device "${DEVICE}" \
        --img-size "${IMG_SIZE}" \
        --batch-size "${BATCH}" \
        ${EXTRA_ARGS}
fi
