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
# Copies best.onnx from RUN_DIR to model/best.onnx so run.py uses the correct weights.
#
# Usage:
#   sbatch scripts/predict_val.slurm
#   sbatch --export=ALL,RUN_DIR=runs/yolo26l_detect scripts/predict_val.slurm
#   sbatch --export=ALL,NO_TTA=1 scripts/predict_val.slurm
#   sbatch --export=ALL,USE_KNN=1 scripts/predict_val.slurm

RUN_DIR=${RUN_DIR:-}
INPUT=${INPUT:-data/yolo_single/val/images}
OUTPUT=${OUTPUT:-output/val_predictions.json}
NO_TTA=${NO_TTA:-0}
USE_KNN=${USE_KNN:-0}
EXTRACT_LAYER=${EXTRACT_LAYER:-}

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

# Copy YOLO ONNX weights to model/best.onnx (same as package.sh uses)
if [[ -n "$RUN_DIR" ]]; then
    BEST_ONNX="${RUN_DIR}/weights/best.onnx"
    if [[ ! -f "$BEST_ONNX" ]]; then
        echo "ERROR: best.onnx not found at ${BEST_ONNX}"
        exit 1
    fi
    cp "$BEST_ONNX" model/best.onnx
    echo "Copied ${BEST_ONNX} → model/best.onnx"
else
    echo "Using existing model/best.onnx"
    if [[ ! -f "model/best.onnx" ]]; then
        echo "ERROR: model/best.onnx not found. Pass RUN_DIR or place best.onnx in model/"
        exit 1
    fi
fi

TTA_FLAG=""
[[ "${NO_TTA}" == "1" ]] && TTA_FLAG="--no-tta"

KNN_FLAG=""
[[ "${USE_KNN}" == "1" ]] && KNN_FLAG="--use-knn"

LAYER_FLAG=""
[[ -n "${EXTRACT_LAYER}" ]] && LAYER_FLAG="--extract-layer ${EXTRACT_LAYER}"

echo "Flags: TTA=${NO_TTA:+OFF}${NO_TTA:-ON} KNN=${USE_KNN}"

if command -v uv >/dev/null 2>&1; then
    uv run python run.py \
        --input "${INPUT}" \
        --output "${OUTPUT}" \
        ${TTA_FLAG} \
        ${KNN_FLAG}
else
    python run.py \
        --input "${INPUT}" \
        --output "${OUTPUT}" \
        ${TTA_FLAG} \
        ${KNN_FLAG}
fi
