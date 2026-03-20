#!/bin/bash
#SBATCH --job-name=nmiai-package-mc
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail

# Packages the YOLO-only multi-class model into submission_yolo_cls.zip.
#
# Usage:
#   sbatch slurm_scripts/package_yolo_cls.slurm
#   sbatch --export=ALL,RUN_DIR=runs/yolov8l_multiclass slurm_scripts/package_yolo_cls.slurm
#
# To run automatically after training:
#   sbatch --dependency=afterok:<train_job_id> slurm_scripts/package_yolo_cls.slurm

RUN_DIR=${RUN_DIR:-}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Packaging YOLO-only submission: run_dir=${RUN_DIR:-auto}"

bash package_yolo_cls.sh ${RUN_DIR:+"$RUN_DIR"}
