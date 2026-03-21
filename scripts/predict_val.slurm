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

# Runs inference using the same model artifacts that package.sh would bundle.
# Copies best.pt from RUN_DIR to model/best.pt so run.py uses the correct weights.
#
# Usage:
#   sbatch scripts/predict_val.slurm
#   sbatch --export=ALL,RUN_DIR=runs/yolov8l_detect11 scripts/predict_val.slurm
#   sbatch --export=ALL,NO_TTA=1 scripts/predict_val.slurm

RUN_DIR=${RUN_DIR:-runs/yolov8l_detect11}
INPUT=${INPUT:-data/yolo_single/val/images}
OUTPUT=${OUTPUT:-output/val_predictions.json}
NO_TTA=${NO_TTA:-0}

mkdir -p "$(dirname "${OUTPUT}")"

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "YOLO run: ${RUN_DIR}"
echo "Running inference: input=${INPUT}, output=${OUTPUT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs from nvidia-smi:"
    nvidia-smi -L
fi

# Copy YOLO weights to model/best.pt (same as package.sh uses)
BEST_PT="${RUN_DIR}/weights/best.pt"
if [[ ! -f "$BEST_PT" ]]; then
    echo "ERROR: best.pt not found at ${BEST_PT}"
    exit 1
fi
cp "$BEST_PT" model/best.pt
echo "Copied ${BEST_PT} → model/best.pt"

TTA_FLAG=""
[[ "${NO_TTA}" == "1" ]] && TTA_FLAG="--no-tta"

if command -v uv >/dev/null 2>&1; then
    uv run python run.py \
        --input "${INPUT}" \
        --output "${OUTPUT}" \
        ${TTA_FLAG}
else
    python run.py \
        --input "${INPUT}" \
        --output "${OUTPUT}" \
        ${TTA_FLAG}
fi
