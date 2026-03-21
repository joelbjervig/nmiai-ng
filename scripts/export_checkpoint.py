"""Export backbone FP16 + cls_head.npy from a mid-training checkpoint.

Allows packaging a submission while training is still running.
Handles both classic and LoRA checkpoints — LoRA adapters are detected
automatically and merged into the base model before export.

Usage:
    python scripts/export_checkpoint.py
    python scripts/export_checkpoint.py --checkpoint model/vit_base_patch14_dinov2_finetune_best.pth
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
COCO_JSON = ROOT / "data" / "train" / "annotations.json"
MODEL_DIR = ROOT / "model"


def has_lora_keys(state_dict: dict) -> bool:
    """Detect if a checkpoint contains LoRA adapter weights."""
    return any("lora_" in k for k in state_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=MODEL_DIR / "vit_base_patch14_dinov2_finetune_best.pth")
    parser.add_argument("--model-name", default="vit_base_patch14_dinov2")
    # LoRA config (must match training if checkpoint is LoRA)
    parser.add_argument("--lora-r",     type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
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

    is_lora = has_lora_keys(ckpt["backbone"])
    print(f"LoRA checkpoint: {is_lora}")

    if is_lora:
        # Reconstruct model with LoRA, load weights, merge
        backbone = timm.create_model(args.model_name, pretrained=False, num_classes=0,
                                      dynamic_img_size=True)
        # Apply same LoRA config as training
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.0,
            bias="none",
        )
        backbone = get_peft_model(backbone, config)
        backbone.load_state_dict(ckpt["backbone"])
        print("Merging LoRA weights into backbone...")
        backbone = backbone.merge_and_unload()
        backbone_state = backbone.state_dict()
    else:
        backbone_state = ckpt["backbone"]

    # Export backbone FP16
    backbone_fp16 = {k: v.half() for k, v in backbone_state.items()}
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
