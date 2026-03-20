"""Quick mAP evaluation against the local val split.

Mirrors the competition scoring:
  Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

Usage:
    python eval_val.py --preds output/val_predictions.json
"""
import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).parent
COCO_JSON = ROOT / "data" / "train" / "annotations.json"
VAL_IMAGES_DIR = ROOT / "data" / "yolo_single" / "val" / "images"


def load_val_image_ids() -> list[int]:
    """Get image IDs for the val split by reading the val images directory."""
    return [
        int(p.stem.split("_", 1)[1])
        for p in sorted(VAL_IMAGES_DIR.glob("img_*.jpg"))
    ]


def make_detection_gt(coco_full: COCO, val_ids: list[int]) -> COCO:
    """Build a single-class COCO GT object (detection only, class-agnostic)."""
    gt = COCO()
    gt.dataset = {
        "images": [coco_full.imgs[i] for i in val_ids if i in coco_full.imgs],
        "annotations": [
            {**ann, "category_id": 1}
            for ann in coco_full.loadAnns(coco_full.getAnnIds(imgIds=val_ids))
        ],
        "categories": [{"id": 1, "name": "product"}],
    }
    gt.createIndex()
    return gt


def make_classification_gt(coco_full: COCO, val_ids: list[int]) -> COCO:
    """Build a multi-class COCO GT object for classification scoring."""
    gt = COCO()
    gt.dataset = {
        "images": [coco_full.imgs[i] for i in val_ids if i in coco_full.imgs],
        "annotations": coco_full.loadAnns(coco_full.getAnnIds(imgIds=val_ids)),
        "categories": list(coco_full.cats.values()),
    }
    gt.createIndex()
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="Path to val_predictions.json")
    args = parser.parse_args()

    preds = json.loads(Path(args.preds).read_text())
    print(f"Loaded {len(preds)} predictions")

    val_ids = load_val_image_ids()
    print(f"Val images: {len(val_ids)}")

    coco_full = COCO(str(COCO_JSON))

    # ── Detection mAP (class-agnostic) ──────────────────────────────────────
    det_gt = make_detection_gt(coco_full, val_ids)
    det_preds = [{"image_id": p["image_id"], "category_id": 1,
                  "bbox": p["bbox"], "score": p["score"]} for p in preds]
    det_dt = det_gt.loadRes(det_preds)
    det_eval = COCOeval(det_gt, det_dt, "bbox")
    det_eval.params.iouThrs = [0.5]
    det_eval.evaluate()
    det_eval.accumulate()
    det_eval.summarize()
    det_map = det_eval.stats[0]

    # ── Classification mAP (correct category required) ───────────────────────
    cls_gt = make_classification_gt(coco_full, val_ids)
    cls_dt = cls_gt.loadRes(preds)
    cls_eval = COCOeval(cls_gt, cls_dt, "bbox")
    cls_eval.params.iouThrs = [0.5]
    cls_eval.evaluate()
    cls_eval.accumulate()
    cls_eval.summarize()
    cls_map = cls_eval.stats[0]

    # ── Final score ───────────────────────────────────────────────────────────
    score = 0.7 * det_map + 0.3 * cls_map
    print(f"\n{'='*45}")
    print(f"  Detection  mAP@0.5 : {det_map:.4f}  (×0.7 = {0.7*det_map:.4f})")
    print(f"  Classif.   mAP@0.5 : {cls_map:.4f}  (×0.3 = {0.3*cls_map:.4f})")
    print(f"  ─────────────────────────────────────────")
    print(f"  Final score        : {score:.4f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
