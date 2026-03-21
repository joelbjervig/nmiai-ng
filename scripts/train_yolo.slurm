#!/bin/bash
#SBATCH --job-name=nmiai-train
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail

# Optional overrides at submit time:
#   sbatch --export=ALL,EPOCHS=100,BATCH=4,IMGSZ=1280,MODEL=yolov8l.pt,DEVICE=0,NAME=exp1 train.slurm
#   sbatch --gres=gpu:1 --export=ALL,DEVICE=0,DATA=yolo_single,NAME=single_class train.slurm
#   sbatch --export=ALL,LR0=0.0012,WEIGHT_DECAY=0.001,FREEZE=12,AUGMENTATION=small_objects,MAX_DET=1000,VAL_CONF=0.1,VAL_IOU=0.55 train.slurm
#   sbatch --export=ALL,DEGREES=3.0,SHEAR=0.5,PERSPECTIVE=0.0008 train.slurm

EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-4}
IMGSZ=${IMGSZ:-1280}
MODEL=${MODEL:-yolov8l.pt}
NAME=${NAME:-yolov8l_detect}
DATA=${DATA:-yolo_single}
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
RECT=${RECT:-False}
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
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting training: model=${MODEL}, data=${DATA}, epochs=${EPOCHS}, batch=${BATCH}, imgsz=${IMGSZ}, device=${DEVICE}, name=${NAME}"
echo "Hyperparams: optimizer=${OPTIMIZER}, lr0=${LR0}, lrf=${LRF}, momentum=${MOMENTUM}, weight_decay=${WEIGHT_DECAY}, warmup_epochs=${WARMUP_EPOCHS}, workers=${WORKERS}, patience=${PATIENCE}, freeze=${FREEZE}, augmentation=${AUGMENTATION}, degrees=${DEGREES}, shear=${SHEAR}, perspective=${PERSPECTIVE}, rect=${RECT}, val_conf=${VAL_CONF}, val_iou=${VAL_IOU}, max_det=${MAX_DET}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs from nvidia-smi:"
    nvidia-smi -L
fi

if command -v uv >/dev/null 2>&1; then
    uv run python train/train_yolo.py \
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
    python train/train_yolo.py \
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
