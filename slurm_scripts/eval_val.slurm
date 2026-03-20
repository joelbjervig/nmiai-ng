#!/bin/bash
#SBATCH --job-name=nmiai-eval
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail

# Optional overrides at submit time:
#   sbatch slurm_scripts/eval_val.slurm
#   sbatch --export=ALL,PREDS=output/val_predictions.json slurm_scripts/eval_val.slurm
#
# To run after predict_val automatically:
#   sbatch --dependency=afterok:<predict_job_id> slurm_scripts/eval_val.slurm

PREDS=${PREDS:-output/val_predictions.json}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Evaluating predictions: ${PREDS}"

if command -v uv >/dev/null 2>&1; then
    uv run python eval_val.py --preds "${PREDS}"
else
    python eval_val.py --preds "${PREDS}"
fi
