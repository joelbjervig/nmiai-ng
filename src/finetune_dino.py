"""Fine-tune DINOv2 on shelf product crops using ArcFace loss.

Strategy
--------
ArcFace loss fine-tunes the embedding space directly for cosine-similarity
nearest-neighbour retrieval (which is what dino_classifier.py uses at
inference). Cross-entropy would work but doesn't optimise the angular
geometry of the embedding space — ArcFace adds an angular margin between
classes, making the NN lookup more reliable.

Only the last --unfreeze-blocks transformer blocks (+ final LayerNorm) are
trained; earlier blocks keep their pretrained features. This prevents
overfitting on the ~22k available crops (~64/class on average).

After training, the backbone is exported as an FP16 state dict that replaces
model/<model_name>_fp16.pth and can be used directly by build_irl_embeddings.py
and dino_classifier.py without any changes.

Usage
-----
    python src/finetune_dino.py
    python src/finetune_dino.py --unfreeze-blocks 6 --epochs 80
    python src/finetune_dino.py --img-size 336 --batch-size 128 --epochs 50
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
        # Simulate bbox imprecision and different zoom levels
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProductCropDataset(Dataset):
    """Yields (crop_tensor, label_idx) from COCO-annotated shelf images.

    category_id → label_idx mapping is built from sorted category IDs so that
    label_idx i always corresponds to the same category_id across runs.

    A small context margin is added around each bbox to give the model a
    sliver of shelf context (helps with boundary cases).
    """

    def __init__(
        self,
        annotations: list[dict],
        image_paths: dict[int, Path],
        catid_to_label: dict[int, int],
        transform,
        min_crop_size: int = 20,
        context_frac: float = 0.05,
    ):
        self.transform     = transform
        self.catid_to_label = catid_to_label
        self.min_crop_size = min_crop_size
        self.context_frac  = context_frac
        self.image_paths   = image_paths

        self.samples: list[tuple[int, list, int]] = []  # (image_id, bbox, label_idx)
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if w < min_crop_size or h < min_crop_size:
                continue
            label = catid_to_label.get(ann["category_id"])
            if label is None:
                continue
            self.samples.append((ann["image_id"], ann["bbox"], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, bbox, label = self.samples[idx]
        img_path = self.image_paths[image_id]

        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        x, y, w, h = bbox

        # Add context margin
        pad_x = w * self.context_frac
        pad_y = h * self.context_frac
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(iw, int(x + w + pad_x))
        y2 = min(ih, int(y + h + pad_y))

        crop = img.crop((x1, y1, x2, y2))
        return self.transform(crop), label


# ── ArcFace head ──────────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """Additive angular margin softmax (ArcFace / InsightFace).

    Optimises the cosine similarity margin between classes directly —
    the same metric used by dino_classifier.py at inference time.

    Args:
        embed_dim:   Dimension of input embeddings (e.g. 768 for ViT-B).
        num_classes: Number of product categories.
        s:           Feature scale (logit temperature). 30–64 typical.
        m:           Angular margin in radians. 0.3–0.5 typical.
    """

    def __init__(self, embed_dim: int, num_classes: int, s: float = 30.0, m: float = 0.4):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)   # cos(π - m)
        self.mm    = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalise both features and class weights → cosine similarity
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))  # (B, C)

        # Target logit: cos(θ + m)  via angle addition formula
        sine         = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-6))
        target_cos   = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        # When θ + m > π (very easy samples), fall back to cos(θ) − m·sin(π − m)
        target_cos   = torch.where(cosine > self.th, target_cos,
                                   cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        logits = one_hot * target_cos + (1.0 - one_hot) * cosine
        return logits * self.s


# ── Backbone helpers ──────────────────────────────────────────────────────────

def freeze_backbone(model, unfreeze_blocks: int):
    """Freeze all parameters, then unfreeze the last N blocks + norm."""
    for param in model.parameters():
        param.requires_grad = False

    total_blocks = len(model.blocks)
    freeze_up_to = total_blocks - unfreeze_blocks

    for i in range(freeze_up_to, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Always unfreeze the final LayerNorm
    for param in model.norm.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {total_blocks} blocks, unfreezing last {unfreeze_blocks} + norm")
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")


def load_backbone(model_name: str, weights_path: Path) -> nn.Module:
    """Load DINOv2 backbone from local FP16 state dict (works offline)."""
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


# ── Training / evaluation ────────────────────────────────────────────────────

def train_one_epoch(
    backbone, head, loader, optimizer, scaler, device, epoch
) -> float:
    backbone.train()
    head.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            features = backbone(imgs)
            logits   = head(features, labels)
            loss     = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(head.parameters()), max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(backbone, head, loader, device) -> tuple[float, float]:
    backbone.eval()
    head.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda"):
            features = backbone(imgs)
            logits   = head(features, labels)
            loss     = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / n, correct / n


def export_fp16(backbone, model_name: str, output_dir: Path):
    """Save backbone as FP16 state dict — drop-in for build_irl_embeddings.py."""
    out_path = output_dir / f"{model_name}_fp16.pth"
    state_dict = {k: v.half() for k, v in backbone.state_dict().items()}
    torch.save(state_dict, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Exported FP16 backbone → {out_path} ({size_mb:.1f} MB)")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_coco_split(val_ratio: float = VAL_RATIO, seed: int = SEED):
    """Load and split COCO annotations, matching prepare_data.py's image split."""
    with open(COCO_JSON) as f:
        coco = json.load(f)

    image_paths = {
        img["id"]: IMAGES_DIR / img["file_name"]
        for img in coco["images"]
    }

    # Same image-level split as prepare_data.py
    rng = random.Random(seed)
    image_ids = sorted(image_paths.keys())
    rng.shuffle(image_ids)
    split_idx  = int(len(image_ids) * (1 - val_ratio))
    train_ids  = set(image_ids[:split_idx])
    val_ids    = set(image_ids[split_idx:])

    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    val_anns   = [a for a in coco["annotations"] if a["image_id"] in val_ids]

    # Build category_id → label_idx (sorted by category_id for reproducibility)
    cat_ids_present = sorted({a["category_id"] for a in coco["annotations"]})
    catid_to_label  = {cid: idx for idx, cid in enumerate(cat_ids_present)}

    print(f"Images: {len(train_ids)} train / {len(val_ids)} val")
    print(f"Annotations: {len(train_anns)} train / {len(val_anns)} val")
    print(f"Categories: {len(catid_to_label)}")

    return train_anns, val_anns, image_paths, catid_to_label


def make_weighted_sampler(dataset: ProductCropDataset) -> WeightedRandomSampler:
    """Over-sample rare classes so each class appears equally per epoch."""
    label_counts: dict[int, int] = defaultdict(int)
    for _, _, label in dataset.samples:
        label_counts[label] += 1

    weights = [1.0 / label_counts[label] for _, _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            default="vit_base_patch14_dinov2")
    parser.add_argument("--unfreeze-blocks",  type=int,   default=4,
                        help="Number of transformer blocks to unfreeze (from the end)")
    parser.add_argument("--epochs",           type=int,   default=60)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--img-size",         type=int,   default=518,
                        help="Crop size fed to DINOv2 (518 = native for patch14)")
    parser.add_argument("--lr-backbone",      type=float, default=2e-5,
                        help="Learning rate for unfrozen backbone blocks")
    parser.add_argument("--lr-head",          type=float, default=1e-4,
                        help="Learning rate for ArcFace head")
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--warmup-epochs",    type=int,   default=5)
    parser.add_argument("--arcface-s",        type=float, default=30.0,
                        help="ArcFace scale (logit temperature)")
    parser.add_argument("--arcface-m",        type=float, default=0.4,
                        help="ArcFace angular margin (radians)")
    parser.add_argument("--min-crop-size",    type=int,   default=20)
    parser.add_argument("--workers",          type=int,   default=4)
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_anns, val_anns, image_paths, catid_to_label = load_coco_split()
    num_classes = len(catid_to_label)

    train_ds = ProductCropDataset(
        train_anns, image_paths, catid_to_label,
        transform=get_train_transform(args.img_size),
        min_crop_size=args.min_crop_size,
    )
    val_ds = ProductCropDataset(
        val_anns, image_paths, catid_to_label,
        transform=get_val_transform(args.img_size),
        min_crop_size=args.min_crop_size,
    )
    print(f"Train crops: {len(train_ds)}  |  Val crops: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_ds),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    weights_path = OUTPUT_DIR / f"{args.model}_fp16.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Backbone weights not found: {weights_path}\n"
            f"  Run: python src/build_embeddings.py --model {args.model}"
        )

    backbone = load_backbone(args.model, weights_path).to(args.device)
    freeze_backbone(backbone, args.unfreeze_blocks)

    embed_dim = backbone.num_features
    head = ArcFaceHead(embed_dim, num_classes,
                       s=args.arcface_s, m=args.arcface_m).to(args.device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone, "lr": args.lr_backbone},
        {"params": head.parameters(),  "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_ckpt     = OUTPUT_DIR / f"{args.model}_finetune_best.pth"

    print(f"\nTraining {args.model}  |  {num_classes} classes  |  "
          f"{args.epochs} epochs  |  device={args.device}")
    print(f"ArcFace: s={args.arcface_s}, m={args.arcface_m}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            backbone, head, train_loader, optimizer, scaler, args.device, epoch
        )
        val_loss, val_acc = evaluate(backbone, head, val_loader, args.device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}  "
            f"lr_bb={optimizer.param_groups[0]['lr']:.2e}  "
            f"({elapsed:.0f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(backbone.state_dict(), best_ckpt)
            print(f"  ↑ new best val acc ({val_acc:.3f}), checkpoint saved")

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print("Loading best checkpoint and exporting FP16 backbone...")

    backbone.load_state_dict(torch.load(best_ckpt, map_location="cpu",
                                        weights_only=True))
    export_fp16(backbone, args.model, OUTPUT_DIR)

    print("\nDone! Next step: re-run build_irl_embeddings.py to rebuild the")
    print("reference embeddings with the fine-tuned backbone.")


if __name__ == "__main__":
    main()
