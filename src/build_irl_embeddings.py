"""Build reference embeddings from real in-the-wild (IRL) training crops.

Instead of clean studio product images, uses ground-truth annotated crops
from the training set. These are in the same domain as the test images
(same lighting, perspective, shelf context), so embedding similarity at
inference time is much more reliable.

Per-category mean embedding is computed across all training crops for that
category, giving a stable centroid with noise averaged out.

Saves to the same ref_embeddings.npz format as build_embeddings.py, so
dino_classifier.py works unchanged.

Usage:
    python src/build_irl_embeddings.py
    python src/build_irl_embeddings.py --model vit_small_patch14_dinov2
    python src/build_irl_embeddings.py --min-crop-size 32
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import timm

ROOT = Path(__file__).resolve().parent.parent
COCO_JSON = ROOT / "data" / "train" / "annotations.json"
IMAGES_DIR = ROOT / "data" / "train" / "images"
OUTPUT_DIR = ROOT / "model"


class PadToSquare:
    """Pad a PIL image to square with neutral gray, preserving aspect ratio."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        max_side = max(w, h)
        padded = Image.new("RGB", (max_side, max_side), (114, 114, 114))
        padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return padded


def get_transform(img_size: int = 518):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_annotations() -> tuple[dict[int, Path], dict[int, list[tuple[int, list]]]]:
    """Parse COCO annotations.

    Returns:
        image_paths: image_id → Path
        cat_to_crops: category_id → [(image_id, [x, y, w, h]), ...]
    """
    with open(COCO_JSON) as f:
        coco = json.load(f)

    image_paths = {
        img["id"]: IMAGES_DIR / img["file_name"]
        for img in coco["images"]
    }

    cat_to_crops: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        cat_to_crops[ann["category_id"]].append((ann["image_id"], ann["bbox"]))

    return image_paths, cat_to_crops


@torch.no_grad()
def embed_crops(
    model,
    crops: list[Image.Image],
    transform,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Embed a list of PIL crops → (N, D) L2-normalised float32 array."""
    all_embs = []
    for i in range(0, len(crops), batch_size):
        batch = [transform(c.convert("RGB")) for c in crops[i:i + batch_size]]
        tensor = torch.stack(batch).to(device)
        features = model(tensor)
        features = F.normalize(features, p=2, dim=-1)
        all_embs.append(features.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def build_irl_lookup(
    model_name: str = "vit_base_patch14_dinov2",
    device: str = "cuda",
    img_size: int = 518,
    batch_size: int = 64,
    min_crop_size: int = 20,
    max_crops_per_cat: int | None = None,
) -> dict:
    print(f"Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=False, num_classes=0)

    # Load FP16 weights exported by build_embeddings.py
    weights_path = OUTPUT_DIR / f"{model_name}_fp16.pth"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    embed_dim = model.num_features
    print(f"Embedding dim: {embed_dim}")

    transform = get_transform(img_size)
    image_paths, cat_to_crops = load_annotations()

    n_cats = len(cat_to_crops)
    n_total = sum(len(v) for v in cat_to_crops.values())
    print(f"Categories: {n_cats}  |  Total annotations: {n_total}")

    embedding_matrix = []
    catid_array = []
    skipped_small = 0

    t0 = time.time()
    for cat_id, crop_list in tqdm(sorted(cat_to_crops.items()), desc="Categories"):
        # Filter degenerate bboxes before loading any images
        valid = [(img_id, bbox) for img_id, bbox in crop_list
                 if bbox[2] >= min_crop_size and bbox[3] >= min_crop_size]
        skipped_small += len(crop_list) - len(valid)

        # Subsample before loading — avoids IO cost for discarded crops
        if max_crops_per_cat is not None and len(valid) > max_crops_per_cat:
            valid = random.sample(valid, max_crops_per_cat)

        crops = []
        for image_id, bbox in valid:
            img_path = image_paths.get(image_id)
            if img_path is None or not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(iw, int(x + w))
            y2 = min(ih, int(y + h))
            crops.append(img.crop((x1, y1, x2, y2)))

        if not crops:
            continue

        embs = embed_crops(model, crops, transform, device, batch_size)  # (N, D)

        # Mean over all crops for this category, then re-normalise
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        embedding_matrix.append(mean_emb.astype(np.float16))
        catid_array.append(cat_id)

    elapsed = time.time() - t0
    print(f"Embedded {n_total - skipped_small} crops in {elapsed:.1f}s "
          f"(skipped {skipped_small} crops smaller than {min_crop_size}px)")

    return {
        "embedding_matrix": np.stack(embedding_matrix),  # (N_cats, D) float16
        "category_ids": np.array(catid_array, dtype=np.int32),  # (N_cats,)
        "embed_dim": embed_dim,
        "model_name": model_name,
        "n_categories": len(catid_array),
        "n_total_crops": n_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_base_patch14_dinov2",
                        help="timm model name (must match the _fp16.pth in model/)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-crop-size", type=int, default=20,
                        help="Skip crops smaller than this in either dimension (px)")
    parser.add_argument("--max-crops-per-cat", type=int, default=None,
                        help="Randomly subsample at most N crops per category (speeds up embedding)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lookup = build_irl_lookup(
        model_name=args.model,
        device=args.device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        min_crop_size=args.min_crop_size,
        max_crops_per_cat=args.max_crops_per_cat,
    )

    out_path = OUTPUT_DIR / "ref_embeddings.npz"
    np.savez_compressed(
        out_path,
        embedding_matrix=lookup["embedding_matrix"],
        category_ids=lookup["category_ids"],
    )

    size_kb = out_path.stat().st_size / 1e3
    print(f"\nSaved: {out_path} ({size_kb:.1f} KB)")
    print(f"  {lookup['n_categories']} categories, {lookup['embed_dim']}d "
          f"(mean over {lookup['n_total_crops']} IRL crops)")


if __name__ == "__main__":
    main()
