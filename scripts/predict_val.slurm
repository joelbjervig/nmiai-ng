#!/bin/bash
#SBATCH --job-name=nmiai-predict
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

# Optional overrides at submit time:
#   sbatch scripts/predict_val.slurm
#   sbatch --export=ALL,INPUT=data/train/images,OUTPUT=output/train_predictions.json scripts/predict_val.slurm

INPUT=${INPUT:-data/yolo_single/val/images}
OUTPUT=${OUTPUT:-output/val_predictions.json}

mkdir -p "$(dirname "${OUTPUT}")"

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Running inference: input=${INPUT}, output=${OUTPUT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs from nvidia-smi:"
    nvidia-smi -L
fi

if command -v uv >/dev/null 2>&1; then
    uv run python run.py \
        --input "${INPUT}" \
        --output "${OUTPUT}"
else
    python run.py \
        --input "${INPUT}" \
        --output "${OUTPUT}"
fi
