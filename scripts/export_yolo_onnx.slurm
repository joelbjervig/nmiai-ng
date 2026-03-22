#!/bin/bash
#SBATCH --job-name=nmiai-export-onnx
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail

# Export YOLO best.pt to ONNX format.
#
# Usage:
#   sbatch --export=ALL,RUN_DIR=runs/yolo26l_detect8 scripts/export_yolo_onnx.slurm

RUN_DIR=${RUN_DIR:-}
IMGSZ=${IMGSZ:-1280}
HALF=${HALF:-1}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"

if [[ -z "$RUN_DIR" ]]; then
    # Auto-detect most recent best.pt
    BEST_PT=$(find runs -name "best.pt" -printf "%T@ %p\n" 2>/dev/null \
              | sort -n | tail -1 | awk '{print $2}')
else
    BEST_PT="${RUN_DIR}/weights/best.pt"
fi

if [[ -z "$BEST_PT" || ! -f "$BEST_PT" ]]; then
    echo "ERROR: best.pt not found at ${BEST_PT:-<auto>}"
    exit 1
fi

echo "Exporting: ${BEST_PT} → ONNX (imgsz=${IMGSZ}, half=${HALF})"

HALF_FLAG=""
[[ "${HALF}" == "1" ]] && HALF_FLAG="half=True"

if command -v uv >/dev/null 2>&1; then
    uv run python -c "
from ultralytics import YOLO
model = YOLO('${BEST_PT}')
model.export(format='onnx', imgsz=${IMGSZ}, opset=17, ${HALF_FLAG})
"
else
    python -c "
from ultralytics import YOLO
model = YOLO('${BEST_PT}')
model.export(format='onnx', imgsz=${IMGSZ}, opset=17, ${HALF_FLAG})
"
fi

ONNX_PATH="${BEST_PT%.pt}.onnx"
if [[ -f "$ONNX_PATH" ]]; then
    echo "Done: ${ONNX_PATH} ($(du -sm "$ONNX_PATH" | awk '{print $1}') MB)"
else
    echo "ERROR: ONNX file not found at ${ONNX_PATH}"
    exit 1
fi
