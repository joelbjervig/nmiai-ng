#!/bin/bash
#SBATCH --job-name=nmiai-export
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail

# Export backbone FP16 + cls_head.npy from a mid-training checkpoint.
#
# Usage:
#   sbatch scripts/export_checkpoint.slurm
#   sbatch --export=ALL,CHECKPOINT=model/vit_base_patch14_dinov2_finetune_best.pth scripts/export_checkpoint.slurm

CHECKPOINT=${CHECKPOINT:-model/vit_base_patch14_dinov2_finetune_best.pth}
MODEL_NAME=${MODEL_NAME:-vit_base_patch14_dinov2}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Exporting from: ${CHECKPOINT}"

if command -v uv >/dev/null 2>&1; then
    uv run python scripts/export_checkpoint.py \
        --checkpoint "${CHECKPOINT}" \
        --model-name "${MODEL_NAME}"
else
    python scripts/export_checkpoint.py \
        --checkpoint "${CHECKPOINT}" \
        --model-name "${MODEL_NAME}"
fi
