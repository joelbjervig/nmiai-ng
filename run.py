"""NorgesGruppen shelf product detection + classification.

Two-stage pipeline with test-time augmentation:
  1. YOLOv8 detection with TTA (multi-scale + flip) + Weighted Boxes Fusion
  2. DINOv2 classification with TTA (horizontal flip logit averaging)

Usage:
    python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

# ── PyTorch 2.6 compatibility ───────────────────────────────────────────────
import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
# ────────────────────────────────────────────────────────────────────────────

from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

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

# TTA settings
DETECT_SCALES = [1280, 1024]  # Multi-scale detection
DETECT_FLIP = True             # Horizontal flip augmentation
WBF_IOU_THR = 0.55            # WBF IoU threshold for merging
WBF_SKIP_BOX_THR = 0.01       # WBF minimum score to keep
USE_CROP_TTA = True            # Flip TTA for classification


def parse_image_id(filename: str) -> int:
    """img_00042.jpg → 42"""
    return int(Path(filename).stem.split("_", 1)[1])


def extract_boxes(result, img_w, img_h):
    """Extract normalized [x1,y1,x2,y2] boxes and scores from YOLO result."""
    if len(result.boxes) == 0:
        return np.zeros((0, 4)), np.array([])
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    # Normalize to [0, 1] for WBF
    boxes_norm = boxes_xyxy.copy()
    boxes_norm[:, [0, 2]] /= img_w
    boxes_norm[:, [1, 3]] /= img_h
    # Clip to [0, 1]
    boxes_norm = np.clip(boxes_norm, 0.0, 1.0)
    return boxes_norm, scores


def detect_with_tta(yolo, img_path, device):
    """Run YOLO at multiple scales + optional flip, merge with WBF."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    all_boxes = []
    all_scores = []
    all_labels = []

    for scale in DETECT_SCALES:
        # Original orientation
        results = yolo(
            str(img_path), imgsz=scale, conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD, max_det=MAX_DET, device=device, verbose=False,
        )
        boxes, scores = extract_boxes(results[0], img_w, img_h)
        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(np.zeros(len(scores)))

        # Horizontal flip
        if DETECT_FLIP:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            results_flip = yolo(
                flipped, imgsz=scale, conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD, max_det=MAX_DET, device=device, verbose=False,
            )
            boxes_flip, scores_flip = extract_boxes(results_flip[0], img_w, img_h)
            if len(boxes_flip) > 0:
                # Unflip x coordinates
                boxes_flip[:, [0, 2]] = 1.0 - boxes_flip[:, [2, 0]]
                all_boxes.append(boxes_flip)
                all_scores.append(scores_flip)
                all_labels.append(np.zeros(len(scores_flip)))

    if not all_boxes:
        return np.zeros((0, 4)), np.array([])

    # Weighted Boxes Fusion
    boxes_fused, scores_fused, _ = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_BOX_THR,
    )

    # Denormalize to pixel coordinates (xyxy)
    boxes_fused[:, [0, 2]] *= img_w
    boxes_fused[:, [1, 3]] *= img_h

    return boxes_fused, scores_fused


def detect_simple(yolo, img_path, device):
    """Single-pass YOLO detection (no TTA)."""
    results = yolo(
        str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
        max_det=MAX_DET, device=device, verbose=False,
    )
    result = results[0]
    if len(result.boxes) == 0:
        return np.zeros((0, 4)), np.array([])
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    return boxes_xyxy, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with img_*.jpg files")
    parser.add_argument("--output", required=True, help="Output predictions.json path")
    parser.add_argument("--no-tta", action="store_true", help="Disable all TTA (faster)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_detect_tta = not args.no_tta and len(DETECT_SCALES) > 0
    use_crop_tta = not args.no_tta and USE_CROP_TTA

    print(f"Device: {device}")
    print(f"Detection TTA: {'ON' if use_detect_tta else 'OFF'} (scales={DETECT_SCALES}, flip={DETECT_FLIP})")
    print(f"Classification TTA: {'ON' if use_crop_tta else 'OFF'}")

    # ── Load models ──────────────────────────────────────────────────────────
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

        # Detection
        if use_detect_tta:
            boxes_xyxy, confs = detect_with_tta(yolo, img_path, device)
        else:
            boxes_xyxy, confs = detect_simple(yolo, img_path, device)

        if len(boxes_xyxy) == 0:
            continue

        # Crop each detection for classification
        img = Image.open(img_path).convert("RGB")
        crops = [
            img.crop((int(x1), int(y1), int(x2), int(y2)))
            for x1, y1, x2, y2 in boxes_xyxy
        ]

        # Classification
        if use_crop_tta:
            classifications = classifier.classify_tta(crops, batch_size=DINO_BATCH_SIZE)
        else:
            classifications = classifier.classify(crops, batch_size=DINO_BATCH_SIZE)

        for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confs, classifications):
            predictions.append({
                "image_id": image_id,
                "category_id": cls["category_id"],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf),
            })

    print(f"Total predictions: {len(predictions)}")

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
