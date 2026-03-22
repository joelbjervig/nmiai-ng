"""NorgesGruppen shelf product detection + classification.

Two-stage pipeline:
  1. YOLO26 detection via ONNX (single pass, no TTA)
  2. DINOv2 classification via linear head (single pass, no TTA)

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

ROOT = Path(__file__).parent
from dino_classifier import DINOClassifier

MODEL_DIR = ROOT / "model"
YOLO_WEIGHTS = MODEL_DIR / "best.onnx"
DINO_WEIGHTS = MODEL_DIR / "vit_base_patch14_dinov2_fp16.pth"
CLS_HEAD = MODEL_DIR / "cls_head.npy"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 1000
DINO_BATCH_SIZE = 64


def parse_image_id(filename: str) -> int:
    return int(Path(filename).stem.split("_", 1)[1])


def letterbox(img: np.ndarray, new_shape: int = 1280):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    pad_h = new_shape - new_h
    pad_w = new_shape - new_w
    top, left = pad_h // 2, pad_w // 2
    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[top:top + new_h, left:left + new_w] = resized
    return padded, scale, left, top


def postprocess_e2e(output: np.ndarray, scale: float, pad_w: int, pad_h: int,
                    img_w: int, img_h: int, conf_thr: float = 0.25):
    """Parse YOLO26 e2e output [1, 300, 6] → (boxes_xyxy, scores)."""
    preds = output[0]  # (300, 6): x1,y1,x2,y2,score,class_id
    scores = preds[:, 4]
    mask = scores > conf_thr
    preds = preds[mask]
    scores = scores[mask]

    if len(preds) == 0:
        return np.zeros((0, 4)), np.array([])

    x1 = np.clip((preds[:, 0] - pad_w) / scale, 0, img_w)
    y1 = np.clip((preds[:, 1] - pad_h) / scale, 0, img_h)
    x2 = np.clip((preds[:, 2] - pad_w) / scale, 0, img_w)
    y2 = np.clip((preds[:, 3] - pad_h) / scale, 0, img_h)

    return np.stack([x1, y1, x2, y2], axis=1), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load YOLO26 ONNX
    print("Loading YOLO ONNX...")
    session = ort.InferenceSession(
        str(YOLO_WEIGHTS),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    print(f"YOLO ready: {session.get_providers()}")

    # Load DINOv2 classifier (linear head)
    print("Loading DINOv2 classifier...")
    classifier = DINOClassifier(
        model_path=DINO_WEIGHTS,
        head_path=CLS_HEAD,
        device=device,
    )

    # Inference
    image_files = sorted(input_dir.glob("img_*.jpg"))
    print(f"Found {len(image_files)} images")

    predictions = []

    for img_path in image_files:
        image_id = parse_image_id(img_path.name)
        img = Image.open(img_path).convert("RGB")
        img_rgb = np.array(img)
        img_h, img_w = img_rgb.shape[:2]

        # Detect
        padded, scale, pad_w, pad_h = letterbox(img_rgb, 1280)
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        output = session.run(None, {input_name: blob})[0]
        boxes_xyxy, confs = postprocess_e2e(output, scale, pad_w, pad_h, img_w, img_h,
                                            conf_thr=CONF_THRESHOLD)

        if len(boxes_xyxy) == 0:
            continue

        # Crop and classify
        crops = [
            img.crop((int(x1), int(y1), int(x2), int(y2)))
            for x1, y1, x2, y2 in boxes_xyxy
        ]
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
