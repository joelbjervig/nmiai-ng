"""Export backbone FP16 + cls_head.npy from a mid-training checkpoint.

Allows packaging a submission while training is still running.

Usage:
    python scripts/export_checkpoint.py
    python scripts/export_checkpoint.py --checkpoint model/vit_base_patch14_dinov2_finetune_best.pth
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
COCO_JSON = ROOT / "data" / "train" / "annotations.json"
MODEL_DIR = ROOT / "model"

VAL_RATIO = 0.15
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=MODEL_DIR / "vit_base_patch14_dinov2_finetune_best.pth")
    parser.add_argument("--model-name", default="vit_base_patch14_dinov2")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return

    # Rebuild catid_to_label mapping (same logic as train_dino.py)
    with open(COCO_JSON) as f:
        coco = json.load(f)
    cat_ids_present = sorted({a["category_id"] for a in coco["annotations"]})
    catid_to_label = {cid: idx for idx, cid in enumerate(cat_ids_present)}
    label_to_catid = {v: k for k, v in catid_to_label.items()}
    print(f"Classes: {len(catid_to_label)}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Export backbone FP16
    backbone_fp16 = {k: v.half() for k, v in ckpt["backbone"].items()}
    backbone_path = MODEL_DIR / f"{args.model_name}_fp16.pth"
    torch.save(backbone_fp16, backbone_path)
    print(f"Exported backbone → {backbone_path} ({backbone_path.stat().st_size / 1e6:.1f} MB)")

    # Export cls_head.npy
    head_data = {
        "weight": ckpt["head"]["weight"].cpu().half().numpy(),
        "bias": ckpt["head"]["bias"].cpu().half().numpy(),
        "label_to_catid": label_to_catid,
    }
    head_path = MODEL_DIR / "cls_head.npy"
    np.save(head_path, head_data, allow_pickle=True)
    print(f"Exported cls_head → {head_path} ({head_path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone. You can now run: bash scripts/package.sh")


if __name__ == "__main__":
    main()
