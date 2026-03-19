"""Convert COCO annotations to YOLO format with train/val split."""
import json
import shutil
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "data" / "train"
COCO_JSON = COCO_DIR / "annotations.json"
COCO_IMAGES = COCO_DIR / "images"
YOLO_DIR = ROOT / "data" / "yolo"
VAL_RATIO = 0.15
SEED = 42


def coco_to_yolo():
    random.seed(SEED)

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

    print(f"Total images: {len(image_ids)}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")

    for split, ids in splits.items():
        img_dir = YOLO_DIR / split / "images"
        lbl_dir = YOLO_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_anns = 0
        for img_id in ids:
            img_info = images[img_id]
            w, h = img_info["width"], img_info["height"]
            fname = img_info["file_name"]

            # Copy image
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
                lines.append(f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")
                n_anns += 1

            label_file.write_text("\n".join(lines) + "\n" if lines else "")

        print(f"{split}: {len(ids)} images, {n_anns} annotations")

    # Write data.yaml
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    names = {c["id"]: c["name"] for c in categories}

    yaml_content = f"""path: {YOLO_DIR}
train: train/images
val: val/images

nc: {len(categories)}
names: {names}
"""
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nWrote {yaml_path}")


if __name__ == "__main__":
    coco_to_yolo()
