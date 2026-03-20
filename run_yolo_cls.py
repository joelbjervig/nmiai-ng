"""NorgesGruppen shelf product detection + classification — YOLO-only variant.

Single-model pipeline: a multi-class YOLOv8 (trained on all 356 product
categories) detects and classifies bounding boxes in one forward pass.

YOLO class index == COCO category_id because prepare_data.py sorts categories
by id (0–355) before writing YOLO labels, so class idx i → category_id i.

Usage:
    python run_yolo_cls.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

# ── PyTorch 2.6 compatibility ────────────────────────────────────────────────
# ultralytics 8.1.0 saves full model pickles; torch 2.6 defaults weights_only=True
import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

# NumPy 2 compatibility: ultralytics 8.1.0 calls np.trapz which was removed
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
# ─────────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO

ROOT = Path(__file__).parent
YOLO_WEIGHTS = ROOT / "model" / "best.pt"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
MAX_DET        = 1000


def parse_image_id(filename: str) -> int:
    """img_00042.jpg → 42"""
    return int(Path(filename).stem.split("_", 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Directory with img_*.jpg files")
    parser.add_argument("--output", required=True, help="Output predictions.json path")
    args = parser.parse_args()

    input_dir   = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading multi-class YOLOv8...")
    yolo = YOLO(str(YOLO_WEIGHTS))

    image_files = sorted(input_dir.glob("img_*.jpg"))
    print(f"Found {len(image_files)} images")

    predictions = []

    for img_path in image_files:
        image_id = parse_image_id(img_path.name)

        results = yolo(
            str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            device=device,
            verbose=False,
        )
        result = results[0]

        if len(result.boxes) == 0:
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()                  # (N, 4)
        confs      = result.boxes.conf.cpu().numpy()                  # (N,)
        cls_ids    = result.boxes.cls.cpu().numpy().astype(int)       # (N,)

        for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
            predictions.append({
                "image_id":   image_id,
                "category_id": int(cls_id),
                # COCO bbox format: [x, y, width, height]
                "bbox":  [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf),
            })

    print(f"Total predictions: {len(predictions)}")

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
