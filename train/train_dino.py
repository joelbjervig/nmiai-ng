"""Fine-tune DINOv2 on shelf product crops with a supervised classification head.

Strategy
--------
Cross-entropy loss with a Linear(768, 356) head directly optimises closed-set
classification — the exact task we need at inference. The last --unfreeze-blocks
transformer blocks (+ final LayerNorm) are fine-tuned while earlier blocks keep
their pretrained features, preventing overfitting on ~22k crops (~64/class).

After training, two artifacts are exported:
  - model/<model_name>_fp16.pth  — backbone state dict (FP16)
  - model/cls_head.npy           — classifier head weights + bias (FP16)

Both are loaded by dino_classifier.py at inference time.

Speed notes
-----------
- Default img_size=518 (37x37=1369 tokens) matches inference resolution.
  518 is the native resolution for patch size 14.
- Crops are pre-extracted to disk (--crop-cache-dir) once, so each epoch
  loads small JPEGs instead of full 4MP+ shelf images.
- --grad-checkpoint trades redundant recomputation for GPU memory, allowing
  larger batch sizes on constrained hardware.

Usage
-----
    python train/train_dino.py
    python train/train_dino.py --img-size 518 --batch-size 32
    python train/train_dino.py --unfreeze-blocks 6 --epochs 80
    python train/train_dino.py --grad-checkpoint --batch-size 64
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

import timm

ROOT       = Path(__file__).resolve().parent.parent
COCO_JSON  = ROOT / "data" / "train" / "annotations.json"
IMAGES_DIR = ROOT / "data" / "train" / "images"
OUTPUT_DIR = ROOT / "model"
CROPS_DIR  = ROOT / "data" / "crops"

VAL_RATIO = 0.15
SEED      = 42


# ── Preprocessing ─────────────────────────────────────────────────────────────

class PadToSquare:
    """Pad a PIL image to square with neutral gray, preserving aspect ratio."""
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        out = Image.new("RGB", (s, s), (114, 114, 114))
        out.paste(img, ((s - w) // 2, (s - h) // 2))
        return out


def get_train_transform(img_size: int):
    return transforms.Compose([
        PadToSquare(),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.6, 1.67),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size: int):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Crop pre-extraction ───────────────────────────────────────────────────────

def extract_crops(
    annotations: list[dict],
    image_paths: dict[int, Path],
    cache_dir: Path,
    min_crop_size: int = 20,
    context_frac: float = 0.05,
) -> None:
    """Extract and save all annotation crops as small JPEGs (one-time cost)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    needed = [
        a for a in annotations
        if not (cache_dir / f"{a['id']}.jpg").exists()
        and a["bbox"][2] >= min_crop_size
        and a["bbox"][3] >= min_crop_size
    ]
    if not needed:
        return

    print(f"Pre-extracting {len(needed)} crops to {cache_dir} ...")
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in needed:
        by_image[ann["image_id"]].append(ann)

    for image_id, anns in tqdm(by_image.items(), desc="Extracting crops"):
        img_path = image_paths.get(image_id)
        if img_path is None or not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        for ann in anns:
            x, y, w, h = ann["bbox"]
            pad_x = w * context_frac
            pad_y = h * context_frac
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(iw, int(x + w + pad_x))
            y2 = min(ih, int(y + h + pad_y))
            crop = img.crop((x1, y1, x2, y2))
            crop.save(cache_dir / f"{ann['id']}.jpg", quality=95)

    print(f"Done. {len(needed)} crops saved.")


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProductCropDataset(Dataset):
    """Yields (crop_tensor, label_idx) for fine-tuning."""

    def __init__(
        self,
        annotations: list[dict],
        image_paths: dict[int, Path],
        catid_to_label: dict[int, int],
        transform,
        min_crop_size: int = 20,
        context_frac: float = 0.05,
        crop_dir: Path | None = None,
    ):
        self.transform      = transform
        self.catid_to_label = catid_to_label
        self.min_crop_size  = min_crop_size
        self.context_frac   = context_frac
        self.image_paths    = image_paths
        self.crop_dir       = crop_dir

        self.samples: list[tuple[int, list, int, int]] = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if w < min_crop_size or h < min_crop_size:
                continue
            label = catid_to_label.get(ann["category_id"])
            if label is None:
                continue
            self.samples.append((ann["image_id"], ann["bbox"], label, ann["id"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, bbox, label, ann_id = self.samples[idx]

        if self.crop_dir is not None:
            crop = Image.open(self.crop_dir / f"{ann_id}.jpg").convert("RGB")
        else:
            img = Image.open(self.image_paths[image_id]).convert("RGB")
            iw, ih = img.size
            x, y, w, h = bbox
            pad_x = w * self.context_frac
            pad_y = h * self.context_frac
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(iw, int(x + w + pad_x))
            y2 = min(ih, int(y + h + pad_y))
            crop = img.crop((x1, y1, x2, y2))

        return self.transform(crop), label


# ── Backbone helpers ──────────────────────────────────────────────────────────

def freeze_backbone(model, unfreeze_blocks: int):
    """Freeze all parameters, then unfreeze the last N blocks + norm."""
    for param in model.parameters():
        param.requires_grad = False

    total_blocks = len(model.blocks)
    for i in range(total_blocks - unfreeze_blocks, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {total_blocks} blocks, unfreezing last {unfreeze_blocks} + norm")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


def load_backbone(model_name: str, weights_path: Path) -> nn.Module:
    """Load DINOv2 backbone from local FP16 state dict (works offline)."""
    model = timm.create_model(model_name, pretrained=False, num_classes=0,
                               dynamic_img_size=True)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


# ── Training / evaluation ────────────────────────────────────────────────────

def train_one_epoch(backbone, head, loader, optimizer, scaler, scheduler, device, epoch,
                    label_smoothing=0.0, dropout=0.0):
    backbone.train()
    head.train()
    total_loss = 0.0
    correct = 0
    n = 0

    bar = tqdm(loader, desc=f"Train {epoch:3d}", leave=True, dynamic_ncols=True)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            features = backbone(imgs)
            if dropout > 0.0:
                features = F.dropout(features, p=dropout, training=True)
            logits   = head(features)
            loss     = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(head.parameters()), max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            correct += (logits.argmax(1) == labels).sum().item()

        total_loss += loss.item() * len(labels)
        n          += len(labels)

        bar.set_postfix(
            loss=f"{total_loss / n:.4f}",
            acc=f"{correct / n:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            gnorm=f"{grad_norm:.2f}",
        )

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for imgs, labels in tqdm(loader, desc="Val      ", leave=True, dynamic_ncols=True):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda"):
            features = backbone(imgs)
            logits   = head(features)
            loss     = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / n, correct / n


def export_fp16(backbone, model_name: str, output_dir: Path):
    """Save backbone as FP16 state dict."""
    out_path = output_dir / f"{model_name}_fp16.pth"
    state_dict = {k: v.half() for k, v in backbone.state_dict().items()}
    torch.save(state_dict, out_path)
    print(f"Exported FP16 backbone → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def export_cls_head(head: nn.Linear, catid_to_label: dict[int, int], output_dir: Path):
    """Save classifier head weights, bias, and label→category_id mapping."""
    head_data = {
        "weight": head.weight.data.cpu().half().numpy(),
        "bias": head.bias.data.cpu().half().numpy(),
        "label_to_catid": {v: k for k, v in catid_to_label.items()},
    }
    out_path = output_dir / "cls_head.npy"
    np.save(out_path, head_data, allow_pickle=True)
    print(f"Exported classifier head → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_coco_split(val_ratio: float = VAL_RATIO, seed: int = SEED):
    with open(COCO_JSON) as f:
        coco = json.load(f)

    image_paths = {img["id"]: IMAGES_DIR / img["file_name"] for img in coco["images"]}

    rng = random.Random(seed)
    image_ids = sorted(image_paths.keys())
    rng.shuffle(image_ids)
    split_idx = int(len(image_ids) * (1 - val_ratio))
    train_ids = set(image_ids[:split_idx])
    val_ids   = set(image_ids[split_idx:])

    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    val_anns   = [a for a in coco["annotations"] if a["image_id"] in val_ids]

    cat_ids_present = sorted({a["category_id"] for a in coco["annotations"]})
    catid_to_label  = {cid: idx for idx, cid in enumerate(cat_ids_present)}

    print(f"Images  : {len(train_ids)} train / {len(val_ids)} val")
    print(f"Crops   : {len(train_anns)} train / {len(val_anns)} val")
    print(f"Classes : {len(catid_to_label)}")
    return train_anns, val_anns, image_paths, catid_to_label


def make_weighted_sampler(dataset: ProductCropDataset) -> WeightedRandomSampler:
    label_counts: dict[int, int] = defaultdict(int)
    for _, _, label, _ in dataset.samples:
        label_counts[label] += 1
    weights = [1.0 / label_counts[label] for _, _, label, _ in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            default="vit_base_patch14_dinov2")
    parser.add_argument("--unfreeze-blocks",  type=int,   default=2)
    parser.add_argument("--epochs",           type=int,   default=60)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--img-size",         type=int,   default=518,
                        help="Training crop size. Must match inference resolution (518).")
    parser.add_argument("--lr-backbone",      type=float, default=2e-5)
    parser.add_argument("--lr-head",          type=float, default=1e-3)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--warmup-epochs",    type=int,   default=5)
    parser.add_argument("--label-smoothing",  type=float, default=0.1,
                        help="Label smoothing for CrossEntropyLoss (0.0 = off)")
    parser.add_argument("--dropout",          type=float, default=0.1,
                        help="Dropout on features before classification head (0.0 = off)")
    parser.add_argument("--min-crop-size",    type=int,   default=20)
    parser.add_argument("--workers",          type=int,   default=4)
    parser.add_argument("--grad-checkpoint",  action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, allows larger batches)")
    parser.add_argument("--crop-cache-dir",   type=Path,  default=CROPS_DIR,
                        help="Directory for pre-extracted crop JPEGs (created on first run)")
    parser.add_argument("--no-crop-cache",    action="store_true",
                        help="Disable crop caching; load from full shelf images (slow)")
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_anns, val_anns, image_paths, catid_to_label = load_coco_split()
    num_classes = len(catid_to_label)

    crop_dir = None
    if not args.no_crop_cache:
        all_anns = train_anns + val_anns
        extract_crops(all_anns, image_paths, args.crop_cache_dir,
                      min_crop_size=args.min_crop_size)
        crop_dir = args.crop_cache_dir

    train_ds = ProductCropDataset(
        train_anns, image_paths, catid_to_label,
        transform=get_train_transform(args.img_size),
        min_crop_size=args.min_crop_size,
        crop_dir=crop_dir,
    )
    val_ds = ProductCropDataset(
        val_anns, image_paths, catid_to_label,
        transform=get_val_transform(args.img_size),
        min_crop_size=args.min_crop_size,
        crop_dir=crop_dir,
    )
    print(f"Dataset : {len(train_ds)} train crops / {len(val_ds)} val crops")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_ds),
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    weights_path = OUTPUT_DIR / f"{args.model}_fp16.pth"
    if not weights_path.exists():
        print(f"Backbone weights not found at {weights_path}, downloading pretrained...")
        backbone = timm.create_model(args.model, pretrained=True, num_classes=0,
                                      dynamic_img_size=True)
        state_dict = {k: v.half() for k, v in backbone.state_dict().items()}
        torch.save(state_dict, weights_path)
        print(f"Saved pretrained backbone → {weights_path}")
    else:
        backbone = load_backbone(args.model, weights_path)

    backbone = backbone.to(args.device)
    freeze_backbone(backbone, args.unfreeze_blocks)

    if args.grad_checkpoint:
        backbone.set_grad_checkpointing(True)
        print("Gradient checkpointing: enabled")

    embed_dim = backbone.num_features
    head = nn.Linear(embed_dim, num_classes).to(args.device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_ckpt    = OUTPUT_DIR / f"{args.model}_finetune_best.pth"

    print(f"\nModel   : {args.model}  ({num_classes} classes, embed_dim={embed_dim})")
    print(f"Head    : Linear({embed_dim}, {num_classes}) + CrossEntropyLoss(label_smoothing={args.label_smoothing})")
    print(f"Dropout : {args.dropout}  |  unfreeze_blocks: {args.unfreeze_blocks}")
    print(f"img_size: {args.img_size}  |  batch: {args.batch_size}  |  epochs: {args.epochs}")
    print(f"crop_cache: {'disabled' if args.no_crop_cache else args.crop_cache_dir}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            backbone, head, train_loader, optimizer, scaler, scheduler, args.device, epoch,
            label_smoothing=args.label_smoothing, dropout=args.dropout,
        )
        val_loss, val_acc = evaluate(backbone, head, val_loader, args.device)

        mem = (torch.cuda.max_memory_allocated(args.device) / 1e9
               if args.device != "cpu" else 0.0)
        elapsed = time.time() - t0

        marker = " ←" if val_acc > best_val_acc else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train {train_loss:.4f}/{train_acc:.3f}  "
            f"val {val_loss:.4f}/{val_acc:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}  "
            f"mem={mem:.1f}GB  {elapsed:.0f}s{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
            }, best_ckpt)

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"\nBest val acc: {best_val_acc:.3f}")
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    export_fp16(backbone, args.model, OUTPUT_DIR)
    export_cls_head(head, catid_to_label, OUTPUT_DIR)


if __name__ == "__main__":
    main()
