#!/usr/bin/env bash
# package.sh — Build submission zip from trained model artifacts.
#
# Run on the HPC after train_yolo and train_dino have completed.
#
# Usage:
#   ./scripts/package.sh                          # auto-detect latest run
#   ./scripts/package.sh runs/yolo26l_detect       # specific run directory
#
# Environment overrides:
#   DINO_MODEL=vit_small_patch14_dinov2 ./scripts/package.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_DIR="${1:-}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch14_dinov2}"
DINO_WEIGHTS="${DINO_WEIGHTS:-}"

MODEL_DIR="$ROOT/model"
STAGING="$ROOT/submission_staging"
OUTPUT_ZIP="$ROOT/submission.zip"
MAX_SIZE_MB=420

echo "=== NorgesGruppen Submission Packager ==="
echo ""

# ── Find YOLO ONNX weights ──────────────────────────────────────────────────
if [[ -n "$RUN_DIR" ]]; then
    BEST_ONNX="$ROOT/$RUN_DIR/weights/best.onnx"
else
    # Auto-detect: most recently modified best.onnx under runs/
    BEST_ONNX=$(find "$ROOT/runs" -name "best.onnx" -printf "%T@ %p\n" 2>/dev/null \
              | sort -n | tail -1 | awk '{print $2}')
fi

# Fallback to model/best.onnx
if [[ -z "$BEST_ONNX" || ! -f "$BEST_ONNX" ]]; then
    BEST_ONNX="$MODEL_DIR/best.onnx"
fi

if [[ ! -f "$BEST_ONNX" ]]; then
    echo "ERROR: best.onnx not found."
    echo "  Train a YOLO model with --export-onnx, or place best.onnx in model/"
    exit 1
fi
echo "YOLO ONNX      : $BEST_ONNX"

# ── Find DINOv2 weights ───────────────────────────────────────────────────────
if [[ -n "$DINO_WEIGHTS" ]]; then
    DINO_PT="$ROOT/$DINO_WEIGHTS"
else
    DINO_PT="$MODEL_DIR/${DINO_MODEL}_fp16.pth"
fi
if [[ ! -f "$DINO_PT" ]]; then
    echo "ERROR: DINOv2 weights not found: $DINO_PT"
    exit 1
fi
echo "DINOv2 weights : $DINO_PT"

# ── Find classifier head weights ─────────────────────────────────────────────
CLS_HEAD="$MODEL_DIR/cls_head.npy"
if [[ ! -f "$CLS_HEAD" ]]; then
    echo "ERROR: Classifier head not found: $CLS_HEAD"
    echo "  Run: sbatch scripts/train_dino.slurm"
    exit 1
fi
echo "Classifier head: $CLS_HEAD"

# ── Build staging directory ───────────────────────────────────────────────────
echo ""
echo "Building staging directory..."
rm -rf "$STAGING"
mkdir -p "$STAGING/model"

cp "$ROOT/run.py"                  "$STAGING/run.py"
cp "$ROOT/dino_classifier.py"      "$STAGING/dino_classifier.py"
cp "$BEST_ONNX"                    "$STAGING/model/best.onnx"
cp "$DINO_PT"                      "$STAGING/model/vit_base_patch14_dinov2_fp16.pth"
cp "$CLS_HEAD"                     "$STAGING/model/cls_head.npy"

# ── Size check ────────────────────────────────────────────────────────────────
yolo_mb=$(du -sm "$STAGING/model/best.onnx" | awk '{print $1}')
dino_mb=$(du -sm "$STAGING/model/vit_base_patch14_dinov2_fp16.pth" | awk '{print $1}')
head_mb=$(du -sm "$STAGING/model/cls_head.npy" | awk '{print $1}')
total_mb=$(du -sm "$STAGING" | awk '{print $1}')

echo ""
echo "Size budget:"
printf "  %-36s %5s MB\n" "best.onnx"                          "$yolo_mb"
printf "  %-36s %5s MB\n" "$(basename "$DINO_PT") → dino"     "$dino_mb"
printf "  %-36s %5s MB\n" "cls_head.npy"                       "$head_mb"
echo "  ─────────────────────────────────────────────"
printf "  %-36s %5s MB  (limit: %s MB)\n" "Total (uncompressed)" "$total_mb" "$MAX_SIZE_MB"

if (( total_mb > MAX_SIZE_MB )); then
    echo ""
    echo "ERROR: Staging exceeds ${MAX_SIZE_MB} MB limit (${total_mb} MB)."
    rm -rf "$STAGING"
    exit 1
fi

# ── Create zip ────────────────────────────────────────────────────────────────
echo ""
echo "Creating $OUTPUT_ZIP ..."
rm -f "$OUTPUT_ZIP"
cd "$STAGING"
python -m zipfile -c "$OUTPUT_ZIP" .
cd "$ROOT"

zip_mb=$(du -sm "$OUTPUT_ZIP" | awk '{print $1}')
echo ""
echo "Done!"
echo "  submission.zip  : ${zip_mb} MB (compressed)"
echo "  Uncompressed    : ${total_mb} MB / ${MAX_SIZE_MB} MB"
echo ""
echo "Upload at: https://app.ainm.no"
