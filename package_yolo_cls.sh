#!/usr/bin/env bash
# package_yolo_cls.sh — Build submission zip for the YOLO-only (multi-class) approach.
#
# Only requires a single weight file (best.pt), so the zip is much smaller
# than the two-stage YOLO+DINOv2 pipeline.
#
# Usage:
#   ./package_yolo_cls.sh                             # auto-detect latest run in runs/
#   ./package_yolo_cls.sh runs/yolov8l_multiclass     # use a specific run directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"

RUN_DIR="${1:-}"

STAGING="$ROOT/submission_staging_yolo_cls"
OUTPUT_ZIP="$ROOT/submission_yolo_cls.zip"
MAX_SIZE_MB=420

echo "=== NorgesGruppen Submission Packager (YOLO multi-class) ==="
echo ""

# ── Find best.pt ──────────────────────────────────────────────────────────────
if [[ -n "$RUN_DIR" ]]; then
    BEST_PT="$ROOT/$RUN_DIR/weights/best.pt"
else
    BEST_PT=$(find "$ROOT/runs" -name "best.pt" -printf "%T@ %p\n" 2>/dev/null \
              | sort -n | tail -1 | awk '{print $2}')
fi

if [[ -z "$BEST_PT" || ! -f "$BEST_PT" ]]; then
    echo "ERROR: best.pt not found."
    echo "  Train a multi-class model first, or pass the run dir:"
    echo "  $0 runs/yolov8l_multiclass"
    exit 1
fi
echo "YOLOv8 weights : $BEST_PT"

# ── Build staging directory ───────────────────────────────────────────────────
echo ""
echo "Building staging directory..."
rm -rf "$STAGING"
mkdir -p "$STAGING/model"

# Submission entry point is always run.py; we copy run_yolo_cls.py as run.py
cp "$ROOT/run_yolo_cls.py"  "$STAGING/run.py"
cp "$BEST_PT"               "$STAGING/model/best.pt"

# ── Size check ────────────────────────────────────────────────────────────────
yolo_mb=$(du -sm "$STAGING/model/best.pt" | awk '{print $1}')
total_mb=$(du -sm "$STAGING" | awk '{print $1}')

echo ""
echo "Size budget:"
printf "  %-36s %5s MB\n" "best.pt" "$yolo_mb"
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
echo "  submission_yolo_cls.zip : ${zip_mb} MB (compressed)"
echo "  Uncompressed            : ${total_mb} MB / ${MAX_SIZE_MB} MB"
echo ""
echo "Upload at: https://app.ainm.no"
