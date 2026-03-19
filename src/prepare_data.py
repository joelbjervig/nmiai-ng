"""Convert COCO annotations to YOLO format with train/val split.

Supports both multi-class (356 product categories) and single-class
(product detection only) modes via --single-class flag.
"""
import argparse
import json
import shutil
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "data" / "train"
COCO_JSON = COCO_DIR / "annotations.json"
COCO_IMAGES = COCO_DIR / "images"
VAL_RATIO = 0.15
SEED = 42


def coco_to_yolo(single_class: bool = False):
    random.seed(SEED)

    yolo_dir = ROOT / "data" / ("yolo_single" if single_class else "yolo")

    with open(COCO_JSON) as f:
        coco = json.load(f)

    # Build image lookup
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Train/val split by image
    image_ids = sorted(images.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * (1 - VAL_RATIO))
    splits = {
        "train": image_ids[:split_idx],
        "val": image_ids[split_idx:],
    }

    mode_str = "SINGLE-CLASS (product)" if single_class else f"MULTI-CLASS (356)"
    print(f"Mode: {mode_str}")
    print(f"Output: {yolo_dir}")
    print(f"Total images: {len(image_ids)}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")

    for split, ids in splits.items():
        img_dir = yolo_dir / split / "images"
        lbl_dir = yolo_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_anns = 0
        for img_id in ids:
            img_info = images[img_id]
            w, h = img_info["width"], img_info["height"]
            fname = img_info["file_name"]

            # Copy image (symlink to save disk space)
            src = COCO_IMAGES / fname
            dst = img_dir / fname
            if not dst.exists():
                shutil.copy2(src, dst)

            # Write YOLO labels
            label_file = lbl_dir / (Path(fname).stem + ".txt")
            lines = []
            for ann in anns_by_image.get(img_id, []):
                # COCO bbox: [x, y, width, height] (top-left corner)
                bx, by, bw, bh = ann["bbox"]
                # YOLO: [class_id, x_center, y_center, width, height] normalized
                x_center = (bx + bw / 2) / w
                y_center = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cls_id = 0 if single_class else ann["category_id"]
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")
                n_anns += 1

            label_file.write_text("\n".join(lines) + "\n" if lines else "")

        print(f"{split}: {len(ids)} images, {n_anns} annotations")

    # Write data.yaml
    if single_class:
        names = {0: "product"}
        nc = 1
    else:
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        names = {c["id"]: c["name"] for c in categories}
        nc = len(categories)

    yaml_content = f"""
        train: train/images
        val: val/images
        nc: {nc}
        names: {names}
        """
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nWrote {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-class", action="store_true",
                        help="Collapse all 356 categories to a single 'product' class")
    args = parser.parse_args()
    coco_to_yolo(single_class=args.single_class)
