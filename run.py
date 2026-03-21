"""NorgesGruppen shelf product detection + classification.

Two-stage pipeline:
  1. YOLOv8 (single-class, fine-tuned) detects bounding boxes.
  2. DINOv2 + linear head classifies each cropped detection.

Usage:
    python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

# ── PyTorch 2.6 compatibility ───────────────────────────────────────────────
# ultralytics 8.1.0 saves full model pickles; torch 2.6 defaults weights_only=True
import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

# NumPy 2 compatibility: ultralytics 8.1.0 calls np.trapz which was removed
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
# ────────────────────────────────────────────────────────────────────────────

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).parent
from dino_classifier import DINOClassifier

MODEL_DIR = ROOT / "model"
YOLO_WEIGHTS = MODEL_DIR / "best.pt"
DINO_WEIGHTS = MODEL_DIR / "vit_base_patch14_dinov2_fp16.pth"
CLS_HEAD = MODEL_DIR / "cls_head.npy"

# Detection hyperparameters
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 1000
DINO_BATCH_SIZE = 64


def parse_image_id(filename: str) -> int:
    """img_00042.jpg → 42"""
    return int(Path(filename).stem.split("_", 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with img_*.jpg files")
    parser.add_argument("--output", required=True, help="Output predictions.json path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load models (once) ──────────────────────────────────────────────────
    print("Loading YOLOv8...")
    yolo = YOLO(str(YOLO_WEIGHTS))

    print("Loading DINOv2 classifier...")
    classifier = DINOClassifier(
        model_path=DINO_WEIGHTS,
        head_path=CLS_HEAD,
        device=device,
    )

    # ── Inference loop ──────────────────────────────────────────────────────
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

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = result.boxes.conf.cpu().numpy()        # (N,)

        # Crop each detection for classification
        img = Image.open(img_path).convert("RGB")
        crops = [
            img.crop((int(x1), int(y1), int(x2), int(y2)))
            for x1, y1, x2, y2 in boxes_xyxy
        ]

        # Classify all crops in one batched pass
        classifications = classifier.classify(crops, batch_size=DINO_BATCH_SIZE)

        for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, classifications):
            predictions.append({
                "image_id": image_id,
                "category_id": cls["category_id"],
                # COCO bbox format: [x, y, width, height]
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                # Detection confidence used for mAP ranking
                "score": float(conf),
            })

    print(f"Total predictions: {len(predictions)}")

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
