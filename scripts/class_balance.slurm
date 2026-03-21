#!/bin/bash
#SBATCH --job-name=nmiai-class-balance
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"

if command -v uv >/dev/null 2>&1; then
    uv run python scripts/class_balance.py
else
    python scripts/class_balance.py
fi
