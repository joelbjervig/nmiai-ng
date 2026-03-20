#!/usr/bin/env bash
# package.sh — Build submission zip from trained model artifacts.
#
# Run on the HPC after training and build_embeddings.py have completed.
#
# Usage:
#   ./package.sh                          # auto-detect latest run in runs/
#   ./package.sh runs/yolov8l_e2e4        # use a specific run directory
#
# Environment overrides:
#   DINO_MODEL=vit_small_patch14_dinov2 ./package.sh   # if you used a smaller model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"

RUN_DIR="${1:-}"
DINO_MODEL="${DINO_MODEL:-vit_base_patch14_dinov2}"

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

# ── Find DINOv2 FP16 weights ─────────────────────────────────────────────────
DINO_PT="$MODEL_DIR/${DINO_MODEL}_fp16.pth"
if [[ ! -f "$DINO_PT" ]]; then
    echo "ERROR: DINOv2 FP16 weights not found: $DINO_PT"
    echo "  Run: python src/build_embeddings.py --model $DINO_MODEL"
    exit 1
fi
echo "DINOv2 weights : $DINO_PT"

# ── Find reference embeddings ────────────────────────────────────────────────
EMBEDDINGS="$MODEL_DIR/ref_embeddings.npy"
CATEGORY_IDS="$MODEL_DIR/category_ids.json"
if [[ ! -f "$EMBEDDINGS" ]]; then
    echo "ERROR: Reference embeddings not found: $EMBEDDINGS"
    echo "  Run: python src/build_embeddings.py"
    exit 1
fi
if [[ ! -f "$CATEGORY_IDS" ]]; then
    echo "ERROR: Category IDs not found: $CATEGORY_IDS"
    echo "  Run: python src/build_embeddings.py"
    exit 1
fi
echo "Embeddings     : $EMBEDDINGS"
echo "Category IDs   : $CATEGORY_IDS"

# ── Build staging directory ───────────────────────────────────────────────────
echo ""
echo "Building staging directory..."
rm -rf "$STAGING"
mkdir -p "$STAGING/model"

cp "$ROOT/run.py"                  "$STAGING/run.py"
cp "$ROOT/src/dino_classifier.py"  "$STAGING/dino_classifier.py"
cp "$BEST_PT"                      "$STAGING/model/best.pt"
cp "$DINO_PT"                      "$STAGING/model/$(basename "$DINO_PT")"
cp "$EMBEDDINGS"                   "$STAGING/model/ref_embeddings.npy"
cp "$CATEGORY_IDS"                 "$STAGING/model/category_ids.json"

# ── Size check ────────────────────────────────────────────────────────────────
yolo_mb=$(du -sm "$STAGING/model/best.pt" | awk '{print $1}')
dino_mb=$(du -sm "$STAGING/model/$(basename "$DINO_PT")" | awk '{print $1}')
emb_mb=$(du -sm  "$STAGING/model/ref_embeddings.npy" | awk '{print $1}')
total_mb=$(du -sm "$STAGING" | awk '{print $1}')

echo ""
echo "Size budget:"
printf "  %-36s %5s MB\n" "best.pt"                     "$yolo_mb"
printf "  %-36s %5s MB\n" "$(basename "$DINO_PT")"      "$dino_mb"
printf "  %-36s %5s MB\n" "ref_embeddings.npy"          "$emb_mb"
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
