"""Train YOLO for single-class product detection (supports YOLOv8 and YOLO26)."""
import argparse
import os
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


def str2bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")



def train(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    data_yaml = ROOT / "data" / args.data / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    model = YOLO(args.model)

    if args.augmentation == "aggressive":
        aug_kwargs = {
            "mosaic": 1.0,
            "close_mosaic": min(20, args.epochs),
            "mixup": 0.15,
            "copy_paste": 0.1,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.4,
            "hsv_v": 0.4,
        }
    elif args.augmentation == "robust":
        # Maximize generalization to unseen stores/lighting/angles
        aug_kwargs = {
            "mosaic": 0.5,
            "close_mosaic": max(10, int(args.epochs * 0.3)),
            "mixup": 0.1,
            "copy_paste": 0.15,
            "degrees": 3.0,
            "shear": 0.3,
            "perspective": 0.0005,
            "translate": 0.05,
            "scale": 0.3,
            "fliplr": 0.5,
            "hsv_h": 0.02,
            "hsv_s": 0.4,
            "hsv_v": 0.4,
        }
    elif args.augmentation == "light":
        aug_kwargs = {
            "mosaic": 0.5,
            "close_mosaic": max(10, int(args.epochs * 0.3)),
            "mixup": 0.05,
            "copy_paste": 0.0,
            "degrees": 2.0,
            "shear": 0.2,
            "perspective": 0.0003,
            "translate": 0.05,
            "scale": 0.2,
            "fliplr": 0.5,
            "hsv_h": 0.01,
            "hsv_s": 0.2,
            "hsv_v": 0.2,
        }
    else:
        aug_kwargs = {
            "mosaic": 0.25,
            "close_mosaic": max(5, int(args.epochs * 0.2)),
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": 3.0,
            "shear": 0.5,
            "perspective": 0.0008,
            "translate": 0.03,
            "scale": 0.1,
            "fliplr": 0.5,
            "hsv_h": 0.01,
            "hsv_s": 0.15,
            "hsv_v": 0.15,
        }

    if args.degrees is not None:
        aug_kwargs["degrees"] = args.degrees
    if args.shear is not None:
        aug_kwargs["shear"] = args.shear
    if args.perspective is not None:
        aug_kwargs["perspective"] = args.perspective

    single_cls = "single" in args.data

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        single_cls=single_cls,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        workers=args.workers,
        patience=args.patience,
        save_period=10,
        freeze=args.freeze,
        rect=args.rect,
        val=True,
        conf=args.val_conf,
        iou=args.val_iou,
        max_det=args.max_det,
        plots=True,
        verbose=True,
        cos_lr=True,
        **aug_kwargs,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26l.pt", help="Model to start from (e.g. yolo26l.pt, yolov8l.pt)")
    parser.add_argument("--data", default="yolo_single", help="Dataset folder under data/ containing data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0", help="cuda device (0, cpu, mps)")
    parser.add_argument("--name", default="yolo26l_detect")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="Export best model to ONNX after training")
    parser.add_argument("--optimizer", default="auto", choices=["SGD", "Adam", "AdamW", "auto"])
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--freeze", type=int, default=0,
                        help="Number of backbone layers to freeze (YOLO26l has 24 blocks)")
    parser.add_argument("--augmentation", default="robust", choices=["small_objects", "robust", "light", "aggressive"])
    parser.add_argument("--degrees", type=float, default=None,
                        help="Override rotation augmentation in degrees")
    parser.add_argument("--shear", type=float, default=None,
                        help="Override shear augmentation in degrees")
    parser.add_argument("--perspective", type=float, default=None,
                        help="Override perspective augmentation strength")
    parser.add_argument("--rect", type=str2bool, default=False,
                        help="Use rectangular batches to preserve image aspect ratio")
    parser.add_argument("--val-conf", type=float, default=0.1,
                        help="Validation confidence threshold for dense scenes")
    parser.add_argument("--val-iou", type=float, default=0.55,
                        help="Validation NMS IoU threshold for dense scenes")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="Maximum detections per image (raise for crowded shelves)")
    
    args = parser.parse_args()

    data_yaml = ROOT / "data" / args.data / "data.yaml"
    print("=" * 60)
    print(f"Training {args.model} on {data_yaml}")
    print(f"Single-class detection mode" if "single" in args.data else "Multi-class mode")
    print(
        f"Optimizer={args.optimizer}, lr0={args.lr0}, weight_decay={args.weight_decay}, augmentation={args.augmentation}, "
        f"degrees_override={args.degrees}, shear_override={args.shear}, perspective_override={args.perspective}, "
        f"rect={args.rect}, val_conf={args.val_conf}, val_iou={args.val_iou}, max_det={args.max_det}"
    )
    print("=" * 60)
    results = train(args)

    if args.export_onnx:
        best_pt = ROOT / "runs" / args.name / "weights" / "best.pt"
        if best_pt.exists():
            print(f"\nExporting {best_pt} to ONNX...")
            best_model = YOLO(str(best_pt))
            onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, opset=17, half=True)
            print(f"ONNX exported → {onnx_path}")
        else:
            print(f"WARNING: best.pt not found at {best_pt}, skipping ONNX export")
