#!/bin/bash
#SBATCH --job-name=nmiai-bbox-stats
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

# Optional overrides:
#   sbatch --export=ALL,DATASET=yolo_single,OUT_JSON=data/yolo_single_bbox_stats.json analyze_bbox_stats.slurm

DATASET=${DATASET:-yolo_single}
OUT_JSON=${OUT_JSON:-data/${DATASET}_bbox_stats.json}

cd "${SLURM_SUBMIT_DIR}"
mkdir -p output

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Computing bbox stats for dataset=${DATASET} -> ${OUT_JSON}"

if command -v uv >/dev/null 2>&1; then
    uv run python src/data/analysis/bbox_stats.py --dataset "${DATASET}" --output "${OUT_JSON}"
else
    python src/data/analysis/bbox_stats.py --dataset "${DATASET}" --output "${OUT_JSON}"
fi
