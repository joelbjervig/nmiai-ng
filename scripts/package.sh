#!/usr/bin/env bash
# package.sh — Build submission zip (max 3 weight files).
#
# Weight files: best.pt (YOLO), _fp16.pth (DINOv2+head), ref_embeddings.npy (kNN)
# Non-weight files: run.py, dino_classifier.py, ref_catids.json
#
# Usage:
#   ./scripts/package.sh                          # auto-detect latest run
#   ./scripts/package.sh runs/yolov8l_detect11    # specific YOLOv8 run

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

# ── Find YOLO best.pt ────────────────────────────────────────────────────────
if [[ -n "$RUN_DIR" ]]; then
    BEST_PT="$ROOT/$RUN_DIR/weights/best.pt"
else
    BEST_PT=$(find "$ROOT/runs" -name "best.pt" -printf "%T@ %p\n" 2>/dev/null \
              | sort -n | tail -1 | awk '{print $2}')
fi
if [[ -z "$BEST_PT" || ! -f "$BEST_PT" ]]; then
    BEST_PT="$MODEL_DIR/best.pt"
fi
if [[ ! -f "$BEST_PT" ]]; then
    echo "ERROR: best.pt not found."
    exit 1
fi
echo "YOLO weights   : $BEST_PT"

# ── Find DINOv2 weights (backbone + head merged) ─────────────────────────────
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

# ── Find kNN embeddings (optional) ───────────────────────────────────────────
REF_EMBEDDINGS="$MODEL_DIR/ref_embeddings.npy"
REF_CATIDS="$MODEL_DIR/ref_catids.json"
HAS_KNN=0
if [[ -f "$REF_EMBEDDINGS" && -f "$REF_CATIDS" ]]; then
    HAS_KNN=1
    echo "Ref embeddings : $REF_EMBEDDINGS"
fi

# ── Build staging directory ───────────────────────────────────────────────────
echo ""
echo "Building staging directory..."
rm -rf "$STAGING"
mkdir -p "$STAGING/model"

cp "$ROOT/run.py"              "$STAGING/run.py"
cp "$ROOT/dino_classifier.py"  "$STAGING/dino_classifier.py"
cp "$BEST_PT"                  "$STAGING/model/best.pt"
cp "$DINO_PT"                  "$STAGING/model/vit_base_patch14_dinov2_fp16.pth"
if (( HAS_KNN )); then
    cp "$REF_EMBEDDINGS"       "$STAGING/model/ref_embeddings.npy"
    cp "$REF_CATIDS"           "$STAGING/model/ref_catids.json"
fi

# ── Size check ────────────────────────────────────────────────────────────────
total_mb=$(du -sm "$STAGING" | awk '{print $1}')
echo ""
echo "Size budget:"
for f in "$STAGING"/model/*; do
    mb=$(du -sm "$f" | awk '{print $1}')
    printf "  %-40s %5s MB\n" "$(basename "$f")" "$mb"
done
echo "  ─────────────────────────────────────────────"
printf "  %-40s %5s MB  (limit: %s MB)\n" "Total (uncompressed)" "$total_mb" "$MAX_SIZE_MB"

# Count weight files
WEIGHT_COUNT=$(find "$STAGING/model" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" \) | wc -l | tr -d ' ')
echo "  Weight files: $WEIGHT_COUNT / 3"

if (( total_mb > MAX_SIZE_MB )); then
    echo "ERROR: Exceeds ${MAX_SIZE_MB} MB limit."
    rm -rf "$STAGING"
    exit 1
fi
if (( WEIGHT_COUNT > 3 )); then
    echo "ERROR: Too many weight files ($WEIGHT_COUNT > 3)."
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
echo "  Weight files    : ${WEIGHT_COUNT} / 3"
echo ""
echo "Upload at: https://app.ainm.no"
