"""Build mean reference embeddings per class from training crops using finetuned DINOv2.

Uses the same crop extraction + preprocessing as train_dino.py for consistency.
Outputs ref_embeddings.npy (N_classes, 768) and ref_catids.json mapping row→category_id.

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --weights model/vit_base_patch14_dinov2_fp16.pth
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import timm

ROOT = Path(__file__).resolve().parent.parent
COCO_JSON = ROOT / "data" / "train" / "annotations.json"
IMAGES_DIR = ROOT / "data" / "train" / "images"
MODEL_DIR = ROOT / "model"
CROPS_DIR = ROOT / "data" / "crops"

SEED = 42


class PadToSquare:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        out = Image.new("RGB", (s, s), (114, 114, 114))
        out.paste(img, ((s - w) // 2, (s - h) // 2))
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="vit_base_patch14_dinov2")
    parser.add_argument("--weights", type=Path,
                        default=MODEL_DIR / "vit_base_patch14_dinov2_fp16.pth")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-crop-size", type=int, default=20)
    parser.add_argument("--max-per-class", type=int, default=10,
                        help="Max crops per class for mean embedding (0 = use all)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Extract from intermediate block (0-11 for ViT-B). None=final CLS token.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Load annotations ──────────────────────────────────────────────────────
    with open(COCO_JSON) as f:
        coco = json.load(f)

    image_paths = {img["id"]: IMAGES_DIR / img["file_name"] for img in coco["images"]}
    cat_ids_present = sorted({a["category_id"] for a in coco["annotations"]})
    catid_to_label = {cid: idx for idx, cid in enumerate(cat_ids_present)}
    label_to_catid = {v: k for k, v in catid_to_label.items()}
    num_classes = len(catid_to_label)

    print(f"Classes: {num_classes}")
    print(f"Total annotations: {len(coco['annotations'])}")

    # ── Group annotations by category ─────────────────────────────────────────
    anns_by_cat: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["bbox"][2] >= args.min_crop_size and ann["bbox"][3] >= args.min_crop_size:
            anns_by_cat[ann["category_id"]].append(ann)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading backbone: {args.weights}")
    model = timm.create_model(args.model_name, pretrained=False, num_classes=0,
                               dynamic_img_size=True)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(args.device).eval().half()

    # Optional: hook for intermediate layer extraction
    intermediate_output = {}
    if args.layer is not None:
        total_blocks = len(model.blocks)
        print(f"Extracting from block {args.layer}/{total_blocks-1}")

        def hook_fn(module, input, output):
            intermediate_output["features"] = output[:, 0, :]  # CLS token

        model.blocks[args.layer].register_forward_hook(hook_fn)

    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((args.img_size, args.img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Extract embeddings per class ──────────────────────────────────────────
    use_cache = CROPS_DIR.exists()
    if use_cache:
        print(f"Using cached crops from {CROPS_DIR}")

    mean_embeddings = torch.zeros(num_classes, model.num_features, device=args.device)
    class_counts = torch.zeros(num_classes, device=args.device)

    for cat_id in tqdm(cat_ids_present, desc="Building embeddings"):
        label = catid_to_label[cat_id]
        anns = anns_by_cat.get(cat_id, [])
        if not anns:
            continue
        if args.max_per_class > 0 and len(anns) > args.max_per_class:
            rng = random.Random(SEED + cat_id)
            anns = rng.sample(anns, args.max_per_class)
        if not anns:
            continue

        # Collect crops
        crops = []
        for ann in anns:
            if use_cache:
                crop_path = CROPS_DIR / f"{ann['id']}.jpg"
                if crop_path.exists():
                    crops.append(Image.open(crop_path).convert("RGB"))
                    continue

            # Fallback: crop from full image
            img_path = image_paths.get(ann["image_id"])
            if img_path is None or not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            x, y, w, h = ann["bbox"]
            pad_x, pad_y = w * 0.05, h * 0.05
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(iw, int(x + w + pad_x))
            y2 = min(ih, int(y + h + pad_y))
            crops.append(img.crop((x1, y1, x2, y2)))

        if not crops:
            continue

        # Embed in batches
        all_embs = []
        for i in range(0, len(crops), args.batch_size):
            batch = crops[i:i + args.batch_size]
            tensors = torch.stack([transform(c) for c in batch]).to(args.device).half()
            with torch.no_grad():
                output = model(tensors)
                features = intermediate_output["features"] if args.layer is not None else output
                features = F.normalize(features, p=2, dim=-1)
            all_embs.append(features)

        cat_embs = torch.cat(all_embs, dim=0)  # (N_crops, 768)
        mean_emb = F.normalize(cat_embs.mean(dim=0, keepdim=True), p=2, dim=-1)
        mean_embeddings[label] = mean_emb.squeeze(0)
        class_counts[label] = len(crops)

    # ── Save ──────────────────────────────────────────────────────────────────
    n_with_embs = int((class_counts > 0).sum().item())
    print(f"\nEmbeddings computed for {n_with_embs}/{num_classes} classes")
    print(f"Min crops/class: {int(class_counts[class_counts > 0].min().item())}")
    print(f"Mean crops/class: {class_counts[class_counts > 0].mean().item():.1f}")

    emb_path = MODEL_DIR / "ref_embeddings.npy"
    np.save(emb_path, mean_embeddings.cpu().float().numpy())
    print(f"Saved embeddings → {emb_path} ({emb_path.stat().st_size / 1e6:.1f} MB)")

    catids_path = MODEL_DIR / "ref_catids.json"
    with open(catids_path, "w") as f:
        json.dump(label_to_catid, f)
    print(f"Saved category mapping → {catids_path}")

    print("\nDone. Use --use-knn flag in run.py to use these embeddings.")


if __name__ == "__main__":
    main()
