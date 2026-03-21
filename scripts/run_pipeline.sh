#!/usr/bin/env bash
# run_pipeline.sh — Submit the full training + eval + package pipeline to SLURM.
#
# Runs on the login node. Submits jobs with dependency chains:
#
#   prepare_data ──┬──> train_yolo ──┬──> predict_val ──> eval_val
#                  └──> train_dino ──┘        │
#                                       package (after yolo + dino)
#
# Usage:
#   ./scripts/run_pipeline.sh                     # full pipeline
#   ./scripts/run_pipeline.sh --skip-prepare       # data already prepared
#   ./scripts/run_pipeline.sh --skip-yolo          # only retrain DINOv2
#   ./scripts/run_pipeline.sh --skip-dino          # only retrain YOLO
#   ./scripts/run_pipeline.sh --skip-prepare --skip-yolo  # just DINOv2 + eval
#
# All SLURM env overrides still work:
#   EPOCHS=100 BATCH=8 ./scripts/run_pipeline.sh
#   YOLO_ARGS="--export=ALL,EPOCHS=50" DINO_ARGS="--export=ALL,IMG_SIZE=518" ./scripts/run_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ── Parse flags ──────────────────────────────────────────────────────────────
SKIP_PREPARE=0
SKIP_YOLO=0
SKIP_DINO=0
SKIP_EVAL=0
SKIP_PACKAGE=0

for arg in "$@"; do
    case "$arg" in
        --skip-prepare)  SKIP_PREPARE=1 ;;
        --skip-yolo)     SKIP_YOLO=1 ;;
        --skip-dino)     SKIP_DINO=1 ;;
        --skip-eval)     SKIP_EVAL=1 ;;
        --skip-package)  SKIP_PACKAGE=1 ;;
        --help|-h)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown flag: $arg (try --help)"
            exit 1 ;;
    esac
done

# Extra sbatch args per job (pass via env vars)
YOLO_ARGS="${YOLO_ARGS:-}"
DINO_ARGS="${DINO_ARGS:-}"

mkdir -p output

echo "=== NorgesGruppen Training Pipeline ==="
echo ""

# Helper: extract job ID from sbatch output ("Submitted batch job 12345")
job_id() { echo "$1" | awk '{print $NF}'; }

DEPS_YOLO=""
DEPS_DINO=""
DEPS_PREDICT=""
DEPS_PACKAGE=""

# ── 1. Prepare data ─────────────────────────────────────────────────────────
if (( SKIP_PREPARE )); then
    echo "[SKIP] prepare_data"
else
    OUT=$(sbatch scripts/prepare_data.slurm)
    PREP_ID=$(job_id "$OUT")
    echo "[SUBMITTED] prepare_data  → job $PREP_ID"
    DEPS_YOLO="--dependency=afterok:${PREP_ID}"
    DEPS_DINO="--dependency=afterok:${PREP_ID}"
fi

# ── 2a. Train YOLO (depends on prepare) ─────────────────────────────────────
if (( SKIP_YOLO )); then
    echo "[SKIP] train_yolo"
else
    OUT=$(sbatch $DEPS_YOLO $YOLO_ARGS scripts/train_yolo.slurm)
    YOLO_ID=$(job_id "$OUT")
    echo "[SUBMITTED] train_yolo   → job $YOLO_ID"
    DEPS_PREDICT="${DEPS_PREDICT:+${DEPS_PREDICT}:}afterok:${YOLO_ID}"
    DEPS_PACKAGE="${DEPS_PACKAGE:+${DEPS_PACKAGE}:}afterok:${YOLO_ID}"
fi

# ── 2b. Train DINOv2 (depends on prepare, parallel with YOLO) ───────────────
if (( SKIP_DINO )); then
    echo "[SKIP] train_dino"
else
    OUT=$(sbatch $DEPS_DINO $DINO_ARGS scripts/train_dino.slurm)
    DINO_ID=$(job_id "$OUT")
    echo "[SUBMITTED] train_dino   → job $DINO_ID"
    DEPS_PREDICT="${DEPS_PREDICT:+${DEPS_PREDICT}:}afterok:${DINO_ID}"
    DEPS_PACKAGE="${DEPS_PACKAGE:+${DEPS_PACKAGE}:}afterok:${DINO_ID}"
fi

# ── 3. Predict on val set (depends on yolo + dino) ──────────────────────────
if [[ -n "$DEPS_PREDICT" ]]; then
    PREDICT_DEP="--dependency=${DEPS_PREDICT}"
else
    PREDICT_DEP=""
fi
OUT=$(sbatch $PREDICT_DEP scripts/predict_val.slurm)
PREDICT_ID=$(job_id "$OUT")
echo "[SUBMITTED] predict_val  → job $PREDICT_ID"

# ── 4. Evaluate (depends on predict) ────────────────────────────────────────
if (( ! SKIP_EVAL )); then
    OUT=$(sbatch --dependency=afterok:${PREDICT_ID} scripts/eval_val.slurm)
    EVAL_ID=$(job_id "$OUT")
    echo "[SUBMITTED] eval_val     → job $EVAL_ID"
fi

# ── 5. Package submission (depends on yolo + dino) ──────────────────────────
if (( ! SKIP_PACKAGE )); then
    if [[ -n "$DEPS_PACKAGE" ]]; then
        PACKAGE_DEP="--dependency=${DEPS_PACKAGE}"
    else
        PACKAGE_DEP=""
    fi
    OUT=$(sbatch $PACKAGE_DEP scripts/package.slurm)
    PACKAGE_ID=$(job_id "$OUT")
    echo "[SUBMITTED] package      → job $PACKAGE_ID"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f output/nmiai-*.out"
