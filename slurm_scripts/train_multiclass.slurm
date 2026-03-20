#!/bin/bash
#SBATCH --job-name=nmiai-train-mc
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail

# Trains a multi-class YOLOv8 model (356 product categories) for the
# YOLO-only detection+classification pipeline.
#
# Requires: data/yolo/ prepared with prepare_data.py (no --single-class flag).
#
# Usage:
#   sbatch slurm_scripts/train_multiclass.slurm
#   sbatch --export=ALL,EPOCHS=200,BATCH=4 slurm_scripts/train_multiclass.slurm
#   sbatch --export=ALL,MODEL=runs/yolov8l_detect/weights/best.pt slurm_scripts/train_multiclass.slurm

EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-4}
IMGSZ=${IMGSZ:-1280}
MODEL=${MODEL:-model/yolov8l.pt}
NAME=${NAME:-yolov8l_multiclass}
DATA=${DATA:-yolo}
OPTIMIZER=${OPTIMIZER:-AdamW}
LR0=${LR0:-0.0012}
LRF=${LRF:-0.01}
MOMENTUM=${MOMENTUM:-0.9}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
WORKERS=${WORKERS:-4}
PATIENCE=${PATIENCE:-60}
FREEZE=${FREEZE:-12}
AUGMENTATION=${AUGMENTATION:-small_objects}
DEGREES=${DEGREES:-1.0}
SHEAR=${SHEAR:-0.1}
PERSPECTIVE=${PERSPECTIVE:-0.0001}
RECT=${RECT:-True}
VAL_CONF=${VAL_CONF:-0.1}
VAL_IOU=${VAL_IOU:-0.55}
MAX_DET=${MAX_DET:-1000}

if [[ -z "${DEVICE:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        visible_gpu_count=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
        if (( visible_gpu_count > 1 )); then
            DEVICE=$(seq -s, 0 $((visible_gpu_count - 1)))
        else
            DEVICE=0
        fi
    else
        DEVICE=0
    fi
fi

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting multi-class training: model=${MODEL}, data=${DATA}, epochs=${EPOCHS}, batch=${BATCH}, imgsz=${IMGSZ}, device=${DEVICE}, name=${NAME}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs:"
    nvidia-smi -L
fi

if command -v uv >/dev/null 2>&1; then
    uv run python src/train.py \
        --model "${MODEL}" \
        --data "${DATA}" \
        --epochs "${EPOCHS}" \
        --batch "${BATCH}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --name "${NAME}" \
        --optimizer "${OPTIMIZER}" \
        --lr0 "${LR0}" \
        --lrf "${LRF}" \
        --momentum "${MOMENTUM}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --workers "${WORKERS}" \
        --patience "${PATIENCE}" \
        --freeze "${FREEZE}" \
        --augmentation "${AUGMENTATION}" \
        --degrees "${DEGREES}" \
        --shear "${SHEAR}" \
        --perspective "${PERSPECTIVE}" \
        --rect "${RECT}" \
        --val-conf "${VAL_CONF}" \
        --val-iou "${VAL_IOU}" \
        --max-det "${MAX_DET}"
else
    python src/train.py \
        --model "${MODEL}" \
        --data "${DATA}" \
        --epochs "${EPOCHS}" \
        --batch "${BATCH}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --name "${NAME}" \
        --optimizer "${OPTIMIZER}" \
        --lr0 "${LR0}" \
        --lrf "${LRF}" \
        --momentum "${MOMENTUM}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --workers "${WORKERS}" \
        --patience "${PATIENCE}" \
        --freeze "${FREEZE}" \
        --augmentation "${AUGMENTATION}" \
        --degrees "${DEGREES}" \
        --shear "${SHEAR}" \
        --perspective "${PERSPECTIVE}" \
        --rect "${RECT}" \
        --val-conf "${VAL_CONF}" \
        --val-iou "${VAL_IOU}" \
        --max-det "${MAX_DET}"
fi
