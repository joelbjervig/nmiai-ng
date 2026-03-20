#!/bin/bash
#SBATCH --job-name=nmiai-prepare-data
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# Optional overrides at submit time:
#   sbatch --export=ALL,SINGLE_CLASS=1 prepare_data.slurm

SINGLE_CLASS=${SINGLE_CLASS:-1}

cd "${SLURM_SUBMIT_DIR}"
mkdir -p output

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Starting data preparation (single_class=${SINGLE_CLASS})"

ARGS=()
if [[ "${SINGLE_CLASS}" == "1" ]]; then
    ARGS+=(--single-class)
fi

if command -v uv >/dev/null 2>&1; then
    uv run python src/prepare_data.py "${ARGS[@]}"
else
    python src/prepare_data.py "${ARGS[@]}"
fi
