"""Print class balance statistics from COCO annotations.

Usage:
    python scripts/class_balance.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COCO_JSON = ROOT / "data" / "train" / "annotations.json"


def main():
    with open(COCO_JSON) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    counts = Counter(a["category_id"] for a in coco["annotations"])

    print(f"Total annotations: {len(coco['annotations'])}")
    print(f"Total categories:  {len(cat_names)}")
    print(f"Categories with annotations: {len(counts)}")
    print()

    # Sort by count descending
    print(f"{'ID':>5}  {'Count':>6}  Name")
    print("-" * 50)
    for cat_id, count in counts.most_common():
        name = cat_names.get(cat_id, "???")
        print(f"{cat_id:>5}  {count:>6}  {name}")

    # Summary stats
    values = list(counts.values())
    print()
    print(f"Min:    {min(values)}")
    print(f"Max:    {max(values)}")
    print(f"Mean:   {sum(values) / len(values):.1f}")
    print(f"Median: {sorted(values)[len(values) // 2]}")

    # Categories with very few samples
    for threshold in [5, 10, 20]:
        n = sum(1 for v in values if v < threshold)
        print(f"Classes with <{threshold} samples: {n}")


if __name__ == "__main__":
    main()
