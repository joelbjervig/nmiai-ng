#!/bin/bash
#SBATCH --job-name=nmiai-geometry-stats
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -euo pipefail

# Optional overrides:
#   sbatch --export=ALL,DATASET=yolo_single,OUT_JSON=data/yolo_single_geometry_stats.json,MAX_IMAGES=10 analyze_geometry_stats.slurm

DATASET=${DATASET:-yolo_single}
OUT_JSON=${OUT_JSON:-data/${DATASET}_geometry_stats.json}
MAX_IMAGES=${MAX_IMAGES:-0}

cd "${SLURM_SUBMIT_DIR}"
mkdir -p output

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Computing geometry stats for dataset=${DATASET}, max_images=${MAX_IMAGES} -> ${OUT_JSON}"

if command -v uv >/dev/null 2>&1; then
    uv run python src/data/analysis/geometry_stats.py --dataset "${DATASET}" --output "${OUT_JSON}" --max-images "${MAX_IMAGES}"
else
    python src/data/analysis/geometry_stats.py --dataset "${DATASET}" --output "${OUT_JSON}" --max-images "${MAX_IMAGES}"
fi
