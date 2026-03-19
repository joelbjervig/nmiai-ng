"""Train YOLOv8 for single-class product detection."""
import argparse
from pathlib import Path

import torch
# ultralytics 8.1.0 uses full model pickle saves which torch 2.6 rejects by default
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "weights_only": False})

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

import numpy as np
# Ultralytics 8.1.0 still calls np.trapz during validation, but NumPy 2 removed it.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid



def train(args):
    data_yaml = ROOT / "data" / args.data / "data.yaml"
    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        # Augmentation — aggressive for small dataset
        mosaic=1.0,
        close_mosaic=20,
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
        workers=8,
        patience=50,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        cos_lr=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/yolov8l.pt", help="Model to start from")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0,1", help="cuda device (0, cpu, mps)")
    parser.add_argument("--name", default="yolov8l_e2e")
    args = parser.parse_args()

    data_yaml = ROOT / "data" / args.data / "data.yaml"
    print("=" * 60)
    print(f"Training {args.model} on {data_yaml}")
    print(f"Single-class detection mode" if "single" in args.data else "Multi-class mode")
    print("=" * 60)
    train(args)
