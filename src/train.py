"""Train YOLOv8l end-to-end and compare against pretrained baseline."""
import argparse
from pathlib import Path

import torch
# ultralytics 8.1.0 uses full model pickle saves which torch 2.6 rejects by default
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "weights_only": False})

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo" / "data.yaml"


def train(args):
    model = YOLO(args.model)

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        # Augmentation — aggressive for small dataset
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        # Training params
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.0005,
        patience=30,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8l.pt", help="Model to start from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="cuda device (0, cpu, mps)")
    parser.add_argument("--name", default="yolov8l_e2e")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Training {args.model} end-to-end on {DATA_YAML}")
    print("=" * 60)
    train(args)
