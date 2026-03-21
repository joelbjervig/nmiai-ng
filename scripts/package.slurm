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
#   sbatch scripts/package.slurm
#   sbatch --export=ALL,RUN_DIR=runs/yolov8l_detect scripts/package.slurm
#   sbatch --export=ALL,DINO_WEIGHTS=model/custom.pth scripts/package.slurm
#
# To run after training automatically:
#   sbatch --dependency=afterok:<train_job_id> scripts/package.slurm

RUN_DIR=${RUN_DIR:-}
export DINO_MODEL=${DINO_MODEL:-vit_base_patch14_dinov2}
export DINO_WEIGHTS=${DINO_WEIGHTS:-}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Packaging submission: run_dir=${RUN_DIR:-auto}, dino_model=${DINO_MODEL}, dino_weights=${DINO_WEIGHTS:-auto}"

bash scripts/package.sh ${RUN_DIR:+"$RUN_DIR"}
