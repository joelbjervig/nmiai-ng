#!/bin/bash
#SBATCH --job-name=nmiai-train
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail

# Optional overrides at submit time:
#   sbatch --export=ALL,EPOCHS=100,BATCH=8,IMGSZ=1280,MODEL=yolov8l.pt,DEVICE=0,1,NAME=exp1 train.slurm
#   sbatch --gres=gpu:1 --export=ALL,DEVICE=0,DATA=yolo,NAME=multiclass train.slurm

EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-16}
IMGSZ=${IMGSZ:-1280}
MODEL=${MODEL:-yolov8l.pt}
NAME=${NAME:-yolov8l_detect}
DATA=${DATA:-yolo_single}

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

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Visible GPUs from nvidia-smi:"
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
        --name "${NAME}"
else
    python src/train.py \
        --model "${MODEL}" \
        --data "${DATA}" \
        --epochs "${EPOCHS}" \
        --batch "${BATCH}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --name "${NAME}"
fi
