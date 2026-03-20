#!/bin/bash
#SBATCH --job-name=nmiai-package
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail

# Packages trained artifacts into submission.zip.
#
# Optional overrides at submit time:
#   sbatch slurm_scripts/package.slurm
#   sbatch --export=ALL,RUN_DIR=runs/yolov8l_e2e4 slurm_scripts/package.slurm
#   sbatch --export=ALL,DINO_WEIGHTS=model/vit_base_patch14_dinov2_finetune_best.pth slurm_scripts/package.slurm
#   sbatch --export=ALL,DINO_MODEL=vit_small_patch14_dinov2 slurm_scripts/package.slurm
#
# To run after training automatically:
#   sbatch --dependency=afterok:<train_job_id> slurm_scripts/package.slurm

RUN_DIR=${RUN_DIR:-runs/yolov8l_detect8}
export DINO_MODEL=${DINO_MODEL:-vit_base_patch14_dinov2}
export DINO_WEIGHTS=${DINO_WEIGHTS:-model/vit_base_patch14_dinov2_finetune_best.pth}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Packaging submission: run_dir=${RUN_DIR:-auto}, dino_model=${DINO_MODEL}, dino_weights=${DINO_WEIGHTS}"

bash package.sh ${RUN_DIR:+"$RUN_DIR"}
