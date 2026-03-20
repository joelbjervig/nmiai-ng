"""Build reference embedding lookup table using DINOv2.

For each product (barcode) with reference images, compute the mean DINOv2
embedding across all available views (main, front, back, left, right, top,
bottom). Save a compact lookup: category_id → mean_embedding.

Also exports the DINOv2 model in FP16 for the submission sandbox.

Usage:
    python src/build_embeddings.py                         # default: vitb14
    python src/build_embeddings.py --model vit_small_patch14_dinov2  # smaller
    python src/build_embeddings.py --export-onnx           # also export ONNX
"""
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import timm

ROOT = Path(__file__).resolve().parent.parent
PRODUCT_DIR = ROOT / "data" / "NM_NGD_product_images"
METADATA = PRODUCT_DIR / "metadata.json"
COCO_JSON = ROOT / "data" / "train" / "annotations.json"
OUTPUT_DIR = ROOT / "model"


def build_barcode_to_catid() -> dict[str, int]:
    """Map product barcode → COCO category_id using product names."""
    with open(METADATA) as f:
        meta = json.load(f)
    with open(COCO_JSON) as f:
        coco = json.load(f)

    name_to_catid = {c["name"]: c["id"] for c in coco["categories"]}

    mapping = {}
    for p in meta["products"] + meta.get("missing", []):
        name = p["product_name"]
        barcode = p["product_code"]
        if name in name_to_catid:
            mapping[barcode] = name_to_catid[name]

    return mapping


def get_transform(img_size: int = 518):
    """DINOv2 canonical preprocessing: 518px for patch14 models (518/14=37 patches)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def embed_images(model, image_paths: list[Path], transform, device: str, batch_size: int = 32) -> np.ndarray:
    """Embed a list of images, return (N, D) array of L2-normalised embeddings."""
    all_embs = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        batch = torch.stack(imgs).to(device)
        features = model(batch)  # (B, D) from timm feature model
        features = F.normalize(features, p=2, dim=-1)
        all_embs.append(features.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def build_lookup(
    model_name: str = "vit_base_patch14_dinov2",
    device: str = "cuda",
    img_size: int = 518,
    batch_size: int = 32,
) -> dict:
    """Build and return the embedding lookup."""
    print(f"Loading {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 → feature extractor
    model = model.to(device).eval()

    embed_dim = model.num_features
    print(f"Embedding dim: {embed_dim}")

    transform = get_transform(img_size)
    barcode_to_catid = build_barcode_to_catid()

    print(f"Barcodes mapped to categories: {len(barcode_to_catid)}")

    # Collect all image paths grouped by category
    catid_to_paths: dict[int, list[Path]] = {}
    catid_to_barcode: dict[int, str] = {}
    for barcode, catid in barcode_to_catid.items():
        product_dir = PRODUCT_DIR / barcode
        if not product_dir.exists():
            continue
        img_paths = sorted(product_dir.glob("*.jpg")) + sorted(product_dir.glob("*.png"))
        if img_paths:
            catid_to_paths[catid] = img_paths
            catid_to_barcode[catid] = barcode

    print(f"Categories with images: {len(catid_to_paths)}")

    # Flatten all images for batch embedding
    all_paths = []
    path_to_catid = {}
    for catid, paths in catid_to_paths.items():
        for p in paths:
            all_paths.append(p)
            path_to_catid[id(p)] = catid
    # Keep proper mapping via index
    flat_catids = [catid for catid, paths in catid_to_paths.items() for _ in paths]

    print(f"Total reference images to embed: {len(all_paths)}")
    t0 = time.time()
    all_embeddings = embed_images(model, all_paths, transform, device, batch_size)
    elapsed = time.time() - t0
    print(f"Embedded {len(all_paths)} images in {elapsed:.1f}s ({len(all_paths)/elapsed:.0f} img/s)")

    # Average embeddings per category, then re-normalise
    catid_embeddings = {}
    idx = 0
    for catid, paths in catid_to_paths.items():
        n = len(paths)
        embs = all_embeddings[idx:idx + n]
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        catid_embeddings[catid] = mean_emb
        idx += n

    # Pack into arrays sorted by catid
    sorted_catids = sorted(catid_embeddings.keys())
    embedding_matrix = np.stack([catid_embeddings[c] for c in sorted_catids]).astype(np.float16)
    catid_array = np.array(sorted_catids, dtype=np.int32)

    return {
        "embedding_matrix": embedding_matrix,  # (N_cats, D) float16
        "category_ids": catid_array,            # (N_cats,) int32
        "embed_dim": embed_dim,
        "model_name": model_name,
        "n_categories": len(sorted_catids),
        "n_total_images": len(all_paths),
    }, model


def export_model_fp16(model, model_name: str, device: str):
    """Save the DINOv2 model weights as FP16 state dict."""
    out_path = OUTPUT_DIR / f"{model_name}_fp16.pth"

    # Convert to FP16 state dict
    state_dict = {k: v.half() for k, v in model.state_dict().items()}
    torch.save(state_dict, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved FP16 model: {out_path} ({size_mb:.1f} MB)")
    return out_path


def export_onnx(model, model_name: str, device: str, img_size: int = 518):
    """Export DINOv2 to ONNX for onnxruntime inference."""
    out_path = OUTPUT_DIR / f"{model_name}.onnx"

    model_fp32 = model.float()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    torch.onnx.export(
        model_fp32,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved ONNX model: {out_path} ({size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_base_patch14_dinov2",
                        help="timm model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=518,
                        help="Image size (518 for DINOv2 patch14)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--export-onnx", action="store_true",
                        help="Also export ONNX model")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lookup, model = build_lookup(
        model_name=args.model,
        device=args.device,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # Save embedding lookup
    lookup_path = OUTPUT_DIR / "ref_embeddings.npz"
    np.savez_compressed(
        lookup_path,
        embedding_matrix=lookup["embedding_matrix"],
        category_ids=lookup["category_ids"],
    )
    size_kb = lookup_path.stat().st_size / 1e3
    print(f"\nSaved lookup: {lookup_path} ({size_kb:.1f} KB)")
    print(f"  Shape: {lookup['embedding_matrix'].shape} ({lookup['n_categories']} categories, {lookup['embed_dim']}d)")
    print(f"  From {lookup['n_total_images']} reference images")

    # Export FP16 model weights
    model_path = export_model_fp16(model, args.model, args.device)

    # Optionally export ONNX
    if args.export_onnx:
        export_onnx(model, args.model, args.device, args.img_size)

    # Print size budget
    yolo_size = (ROOT / "model" / "yolov8l.pt").stat().st_size / 1e6
    dino_size = model_path.stat().st_size / 1e6
    lookup_size = lookup_path.stat().st_size / 1e6
    total = yolo_size + dino_size + lookup_size

    print(f"\n{'='*50}")
    print(f"SIZE BUDGET")
    print(f"  YOLOv8l:          {yolo_size:>8.1f} MB")
    print(f"  DINOv2 FP16:      {dino_size:>8.1f} MB")
    print(f"  Ref embeddings:   {lookup_size:>8.1f} MB")
    print(f"  ─────────────────────────")
    print(f"  Total:            {total:>8.1f} MB")
    print(f"  Budget:           {420.0:>8.1f} MB")
    print(f"  Remaining:        {420.0 - total:>8.1f} MB")
    print(f"  {'✓ FITS' if total <= 420 else '✗ OVER BUDGET'}")


if __name__ == "__main__":
    main()
