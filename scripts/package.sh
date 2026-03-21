#!/usr/bin/env bash
# package.sh — Build submission zip from trained model artifacts.
#
# Run on the HPC after train_yolo and train_dino have completed.
#
# Usage:
#   ./package.sh                          # auto-detect latest run in runs/
#   ./package.sh runs/yolov8l_e2e4        # use a specific run directory
#
# Environment overrides:
#   DINO_MODEL=vit_small_patch14_dinov2 ./package.sh   # if you used a smaller model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_DIR="${1:-}"
RUN_DIR=${RUN_DIR:-runs/yolov8l_detect11}

DINO_MODEL="${DINO_MODEL:-vit_base_patch14_dinov2}"
# DINO_WEIGHTS overrides the auto-derived path — use this for fine-tuned models:
#   DINO_WEIGHTS=model/vit_base_patch14_dinov2_finetune_best.pth ./package.sh
DINO_WEIGHTS="${DINO_WEIGHTS:-}"

MODEL_DIR="$ROOT/model"
STAGING="$ROOT/submission_staging"
OUTPUT_ZIP="$ROOT/submission.zip"
MAX_SIZE_MB=420

echo "=== NorgesGruppen Submission Packager ==="
echo ""

# ── Find best.pt ─────────────────────────────────────────────────────────────
if [[ -n "$RUN_DIR" ]]; then
    BEST_PT="$ROOT/$RUN_DIR/weights/best.pt"
else
    # Auto-detect: most recently modified best.pt under runs/
    BEST_PT=$(find "$ROOT/runs" -name "best.pt" -printf "%T@ %p\n" 2>/dev/null \
              | sort -n | tail -1 | awk '{print $2}')
fi

if [[ -z "$BEST_PT" || ! -f "$BEST_PT" ]]; then
    echo "ERROR: best.pt not found."
    echo "  Train a model first, or pass the run dir as an argument:"
    echo "  $0 runs/yolov8l_e2e4"
    exit 1
fi
echo "YOLOv8 weights : $BEST_PT"

# ── Find DINOv2 weights ───────────────────────────────────────────────────────
if [[ -n "$DINO_WEIGHTS" ]]; then
    DINO_PT="$ROOT/$DINO_WEIGHTS"
else
    DINO_PT="$MODEL_DIR/${DINO_MODEL}_fp16.pth"
fi
if [[ ! -f "$DINO_PT" ]]; then
    echo "ERROR: DINOv2 weights not found: $DINO_PT"
    echo "  Default path: model/${DINO_MODEL}_fp16.pth"
    echo "  Or override:  DINO_WEIGHTS=model/vit_base_patch14_dinov2_finetune_best.pth $0"
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
cp "$BEST_PT"                      "$STAGING/model/best.pt"
cp "$DINO_PT"                      "$STAGING/model/vit_base_patch14_dinov2_fp16.pth"
cp "$CLS_HEAD"                     "$STAGING/model/cls_head.npy"

# ── Size check ────────────────────────────────────────────────────────────────
yolo_mb=$(du -sm "$STAGING/model/best.pt" | awk '{print $1}')
dino_mb=$(du -sm "$STAGING/model/vit_base_patch14_dinov2_fp16.pth" | awk '{print $1}')
head_mb=$(du -sm "$STAGING/model/cls_head.npy" | awk '{print $1}')
total_mb=$(du -sm "$STAGING" | awk '{print $1}')

echo ""
echo "Size budget:"
printf "  %-36s %5s MB\n" "best.pt"                          "$yolo_mb"
printf "  %-36s %5s MB\n" "$(basename "$DINO_PT") → dino"   "$dino_mb"
printf "  %-36s %5s MB\n" "cls_head.npy"                     "$head_mb"
echo "  ─────────────────────────────────────────────"
printf "  %-36s %5s MB  (limit: %s MB)\n" "Total (uncompressed)" "$total_mb" "$MAX_SIZE_MB"

if (( total_mb > MAX_SIZE_MB )); then
    echo ""
    echo "ERROR: Staging exceeds ${MAX_SIZE_MB} MB limit (${total_mb} MB)."
    echo "  Try a smaller DINOv2 variant:"
    echo "  DINO_MODEL=vit_small_patch14_dinov2 $0"
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
