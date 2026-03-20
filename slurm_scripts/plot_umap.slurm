#!/bin/bash
#SBATCH --job-name=nmiai-umap
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

# Plots a UMAP projection of the reference embeddings coloured by category.
# Output saved to output/umap_embeddings.png (or OUTPUT path).
#
# Usage:
#   sbatch slurm_scripts/plot_umap.slurm
#   sbatch --export=ALL,OUTPUT=output/umap_irl.png slurm_scripts/plot_umap.slurm

NEIGHBORS=${NEIGHBORS:-15}
MIN_DIST=${MIN_DIST:-0.1}
OUTPUT=${OUTPUT:-output/umap_embeddings.png}
DPI=${DPI:-150}
POINT_SIZE=${POINT_SIZE:-6}

cd "${SLURM_SUBMIT_DIR}"

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"

# ── Install umap-learn if not already available ───────────────────────────────
if command -v uv >/dev/null 2>&1; then
    uv run python -c "import umap" 2>/dev/null || uv add umap-learn
else
    python -c "import umap" 2>/dev/null || pip install --quiet umap-learn
fi

echo "Plotting UMAP: neighbors=${NEIGHBORS}, min_dist=${MIN_DIST}, output=${OUTPUT}"

if command -v uv >/dev/null 2>&1; then
    uv run python src/plot_umap.py \
        --neighbors "${NEIGHBORS}" \
        --min-dist  "${MIN_DIST}" \
        --output    "${OUTPUT}" \
        --dpi       "${DPI}" \
        --point-size "${POINT_SIZE}"
else
    python src/plot_umap.py \
        --neighbors "${NEIGHBORS}" \
        --min-dist  "${MIN_DIST}" \
        --output    "${OUTPUT}" \
        --dpi       "${DPI}" \
        --point-size "${POINT_SIZE}"
fi
