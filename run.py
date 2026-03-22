"""NorgesGruppen shelf product detection + classification.

Two-stage pipeline with test-time augmentation:
  1. YOLO detection via ONNX (multi-scale + flip) + Weighted Boxes Fusion
  2. DINOv2 classification with TTA (horizontal flip logit averaging)

Uses onnxruntime-gpu for YOLO inference (sandbox-compatible, no ultralytics needed).

Usage:
    python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).parent
from dino_classifier import DINOClassifier

MODEL_DIR = ROOT / "model"
YOLO_WEIGHTS = MODEL_DIR / "best.onnx"
DINO_WEIGHTS = MODEL_DIR / "vit_base_patch14_dinov2_fp16.pth"
CLS_HEAD = MODEL_DIR / "cls_head.npy"

# Detection hyperparameters
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 1000
DINO_BATCH_SIZE = 64

# TTA settings
DETECT_SCALES = [1280]
DETECT_FLIP = True
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.01
USE_CROP_TTA = True


def parse_image_id(filename: str) -> int:
    """img_00042.jpg → 42"""
    return int(Path(filename).stem.split("_", 1)[1])


# ── ONNX YOLO inference ────────────────────────────────────────────────────

def letterbox(img: np.ndarray, new_shape: int = 1280):
    """Resize image with letterbox padding (preserve aspect ratio).

    Returns padded image, scale factor, and (pad_w, pad_h).
    """
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

    pad_h = new_shape - new_h
    pad_w = new_shape - new_w
    top, left = pad_h // 2, pad_w // 2

    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[top:top + new_h, left:left + new_w] = resized

    return padded, scale, (left, top)


def preprocess(img: np.ndarray, imgsz: int = 1280):
    """Preprocess image for YOLO ONNX: letterbox, normalize, CHW, batch."""
    padded, scale, (pad_w, pad_h) = letterbox(img, imgsz)
    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
    return blob, scale, pad_w, pad_h


def postprocess(output: np.ndarray, scale: float, pad_w: int, pad_h: int,
                img_w: int, img_h: int, conf_thr: float = 0.25, iou_thr: float = 0.45):
    """Parse YOLO ONNX output → (boxes_xyxy, scores) in original image coords.

    Handles both formats:
      - YOLOv8: [1, 4+nc, N] where rows 0-3 = cx,cy,w,h, row 4+ = class scores
      - YOLO26 e2e: [1, N, 6] where cols = x1,y1,x2,y2,score,class_id
    """
    print(f"  ONNX output shape: {output.shape}", flush=True)

    # Detect format based on shape
    if output.ndim == 3 and output.shape[2] == 6:
        # YOLO26 end-to-end format: [1, N, 6] = x1,y1,x2,y2,score,class_id
        preds = output[0]  # (N, 6)
        scores = preds[:, 4]
        mask = scores > conf_thr
        preds = preds[mask]
        scores = scores[mask]

        if len(preds) == 0:
            return np.zeros((0, 4)), np.array([])

        # Already x1,y1,x2,y2 — just rescale from letterbox to original
        x1 = (preds[:, 0] - pad_w) / scale
        y1 = (preds[:, 1] - pad_h) / scale
        x2 = (preds[:, 2] - pad_w) / scale
        y2 = (preds[:, 3] - pad_h) / scale
    else:
        # YOLOv8 format: [1, 4+nc, N] → transpose to [N, 4+nc]
        preds = output[0].T

        scores = preds[:, 4:].max(axis=1)
        mask = scores > conf_thr
        preds = preds[mask]
        scores = scores[mask]

        if len(preds) == 0:
            return np.zeros((0, 4)), np.array([])

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - w / 2 - pad_w) / scale
        y1 = (cy - h / 2 - pad_h) / scale
        x2 = (cx + w / 2 - pad_w) / scale
        y2 = (cy + h / 2 - pad_h) / scale
    mask = scores > conf_thr
    preds = preds[mask]
    scores = scores[mask]

    if len(preds) == 0:
        return np.zeros((0, 4)), np.array([])

    # Convert cx, cy, w, h → x1, y1, x2, y2
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Remove padding and rescale to original image coordinates
    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    # Clip to image bounds
    x1 = np.clip(x1, 0, img_w)
    y1 = np.clip(y1, 0, img_h)
    x2 = np.clip(x2, 0, img_w)
    y2 = np.clip(y2, 0, img_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep][:MAX_DET], scores[keep][:MAX_DET]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Standard greedy NMS."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= iou_thr)[0]
        order = order[remaining + 1]

    return keep


class YOLODetector:
    """YOLO ONNX detector with optional TTA (multi-scale + flip + WBF)."""

    def __init__(self, model_path: Path):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"YOLO ONNX ready: {model_path.name}, providers={self.session.get_providers()}")

    def _run_single(self, img_rgb: np.ndarray, imgsz: int):
        """Run YOLO on a single image at a given scale. Returns (boxes_xyxy, scores)."""
        img_h, img_w = img_rgb.shape[:2]
        blob, scale, pad_w, pad_h = preprocess(img_rgb, imgsz)
        output = self.session.run(None, {self.input_name: blob})[0]
        return postprocess(output, scale, pad_w, pad_h, img_w, img_h,
                           conf_thr=CONF_THRESHOLD, iou_thr=IOU_THRESHOLD)

    def detect(self, img_rgb: np.ndarray):
        """Single-pass detection at default scale."""
        return self._run_single(img_rgb, DETECT_SCALES[0] if DETECT_SCALES else 1280)

    def detect_tta(self, img_rgb: np.ndarray):
        """Multi-scale + flip TTA with Weighted Boxes Fusion."""
        img_h, img_w = img_rgb.shape[:2]

        all_boxes = []
        all_scores = []
        all_labels = []

        for scale in DETECT_SCALES:
            # Original
            boxes, scores = self._run_single(img_rgb, scale)
            if len(boxes) > 0:
                boxes_norm = boxes.copy()
                boxes_norm[:, [0, 2]] /= img_w
                boxes_norm[:, [1, 3]] /= img_h
                boxes_norm = np.clip(boxes_norm, 0.0, 1.0)
                all_boxes.append(boxes_norm)
                all_scores.append(scores)
                all_labels.append(np.zeros(len(scores)))

            # Horizontal flip
            if DETECT_FLIP:
                flipped = img_rgb[:, ::-1, :].copy()
                boxes_f, scores_f = self._run_single(flipped, scale)
                if len(boxes_f) > 0:
                    boxes_f_norm = boxes_f.copy()
                    boxes_f_norm[:, [0, 2]] /= img_w
                    boxes_f_norm[:, [1, 3]] /= img_h
                    boxes_f_norm = np.clip(boxes_f_norm, 0.0, 1.0)
                    # Unflip x coordinates
                    boxes_f_norm[:, [0, 2]] = 1.0 - boxes_f_norm[:, [2, 0]]
                    all_boxes.append(boxes_f_norm)
                    all_scores.append(scores_f)
                    all_labels.append(np.zeros(len(scores_f)))

        if not all_boxes:
            return np.zeros((0, 4)), np.array([])

        boxes_fused, scores_fused, _ = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=WBF_IOU_THR,
            skip_box_thr=WBF_SKIP_BOX_THR,
        )

        # Denormalize
        boxes_fused[:, [0, 2]] *= img_w
        boxes_fused[:, [1, 3]] *= img_h

        return boxes_fused, scores_fused


# ── Main ────────────────────────────────────────────────────────────────────

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
    use_detect_tta = not args.no_tta and len(DETECT_SCALES) > 1
    use_crop_tta = not args.no_tta and USE_CROP_TTA

    print(f"Device: {device}")
    print(f"Detection TTA: {'ON' if use_detect_tta else 'OFF'} (scales={DETECT_SCALES}, flip={DETECT_FLIP})")
    print(f"Classification TTA: {'ON' if use_crop_tta else 'OFF'}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading YOLO ONNX...")
    detector = YOLODetector(YOLO_WEIGHTS)

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

        img = Image.open(img_path).convert("RGB")
        img_rgb = np.array(img)

        # Detection
        if use_detect_tta:
            boxes_xyxy, confs = detector.detect_tta(img_rgb)
        else:
            boxes_xyxy, confs = detector.detect(img_rgb)

        if len(boxes_xyxy) == 0:
            continue

        # Crop each detection for classification
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
                "score": float(conf * cls["score"]),
            })

    print(f"Total predictions: {len(predictions)}")

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
