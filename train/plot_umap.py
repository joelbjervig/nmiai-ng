"""Plot a UMAP projection of the reference embeddings, coloured by category.

Useful for visually inspecting how well the embedding space separates products.
Tight clusters = discriminative embeddings. Overlapping clusters = confusion.

Usage:
    python src/plot_umap.py
    python src/plot_umap.py --output output/umap.png
    python src/plot_umap.py --neighbors 30 --min-dist 0.05
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT       = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT / "model"
OUTPUT_DIR = ROOT / "output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings",  type=Path, default=MODEL_DIR / "ref_embeddings.npy")
    parser.add_argument("--cat-ids",     type=Path, default=MODEL_DIR / "category_ids.json")
    parser.add_argument("--output",      type=Path, default=OUTPUT_DIR / "umap_embeddings.png")
    parser.add_argument("--neighbors",   type=int,  default=15,  help="UMAP n_neighbors")
    parser.add_argument("--min-dist",    type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--dpi",         type=int,  default=150)
    parser.add_argument("--point-size",  type=float, default=6.0)
    args = parser.parse_args()

    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    print(f"Loading embeddings from {args.embeddings} ...")
    embeddings = np.load(args.embeddings).astype(np.float32)  # (N, D)

    with open(args.cat_ids) as f:
        cat_ids = np.array(json.load(f), dtype=np.int32)      # (N,)

    n_points, dim = embeddings.shape
    n_cats = len(np.unique(cat_ids))
    print(f"  {n_points} embeddings  |  {dim}d  |  {n_cats} categories")

    print(f"Running UMAP (n_neighbors={args.neighbors}, min_dist={args.min_dist}) ...")
    reducer = umap.UMAP(
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)   # (N, 2)

    # Map category IDs to a continuous colour in [0, 1]
    unique_cats = np.unique(cat_ids)
    cat_to_rank = {c: i for i, c in enumerate(unique_cats)}
    colours = np.array([cat_to_rank[c] for c in cat_ids], dtype=float)
    colours /= max(len(unique_cats) - 1, 1)

    print("Plotting ...")
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colours,
        cmap="hsv",
        s=args.point_size,
        alpha=0.6,
        linewidths=0,
    )
    ax.set_title(f"UMAP of reference embeddings  ({n_points} points, {n_cats} categories)",
                 fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    # Colourbar as a rough category index guide (no per-class labels)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Category index (0 → 355)", fontsize=9)
    cbar.set_ticks([])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
