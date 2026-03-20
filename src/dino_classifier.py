"""DINOv2 embedding-based product classifier for inference.

Loads the FP16 DINOv2 model and pre-computed reference embeddings, then
classifies cropped product images via cosine similarity lookup.

Reference embeddings are stored per-view (front, back, left, right, etc.).
Classification uses max cosine similarity across all views of a category,
which is more discriminative than averaging views into a single vector.

Crops are padded to square before resizing to preserve aspect ratio.

Designed to work within the competition sandbox (no os module, uses pathlib).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import timm


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


class DINOClassifier:
    """Two-step classifier: embed crop → nearest-neighbour in reference table.

    Reference embeddings are stored per-view (not averaged per category).
    Classification uses max cosine similarity across all views of a category.
    """

    def __init__(
        self,
        model_path: Path,
        embeddings_path: Path,
        model_name: str = "vit_base_patch14_dinov2",
        img_size: int = 518,
        device: str = "cuda",
    ):
        self.device = device
        self.img_size = img_size

        # Load DINOv2 model from FP16 weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = {k: v.float() for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval().half()

        self.embed_dim = self.model.num_features

        # Load per-view reference embeddings
        # embedding_matrix: (N_views, D) — one row per reference image
        # category_ids:     (N_views,)   — category ID for each row
        data = np.load(embeddings_path)
        self.ref_embeddings = torch.from_numpy(
            data["embedding_matrix"].astype(np.float32)
        ).half().to(device)

        # Pre-compute unique categories and view→category-index mapping
        # for efficient scatter_reduce max-pooling at query time
        cat_ids_tensor = torch.from_numpy(data["category_ids"].astype(np.int64))
        self.unique_cats, self.view_to_cat_idx = torch.unique(
            cat_ids_tensor, sorted=True, return_inverse=True
        )
        self.view_to_cat_idx = self.view_to_cat_idx.to(device)
        self.n_cats = len(self.unique_cats)

        # Aspect-ratio-preserving preprocessing: pad to square, then resize
        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"DINOClassifier ready: {model_name}, {self.n_cats} categories "
              f"({len(self.ref_embeddings)} views), device={device}")

    @torch.no_grad()
    def embed_crops(self, crops: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Embed a list of PIL crops → (N, D) normalised FP16 tensor."""
        all_embs = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            tensors = [self.transform(c.convert("RGB")) for c in batch_crops]
            batch = torch.stack(tensors).to(self.device).half()
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=-1)
            all_embs.append(features)
        return torch.cat(all_embs, dim=0)

    def classify(self, crops: list[Image.Image], batch_size: int = 64) -> list[dict]:
        """Classify cropped product images.

        Computes cosine similarity against all per-view reference embeddings,
        then max-pools over views to get the best score per category.

        Returns list of dicts with keys:
            - category_id: int (COCO category ID, 0-355)
            - score: float (max cosine similarity across views, 0-1)
        """
        if not crops:
            return []

        embeddings = self.embed_crops(crops, batch_size)  # (N, D)
        N = len(crops)

        # Cosine similarity against all per-view reference embeddings
        similarities = embeddings @ self.ref_embeddings.T  # (N, N_views)

        # Max-pool over views per category
        # cat_sims[i, j] = max similarity of query i over all views of category j
        idx = self.view_to_cat_idx.unsqueeze(0).expand(N, -1)  # (N, N_views)
        cat_sims = torch.full(
            (N, self.n_cats), float("-inf"), device=self.device, dtype=torch.float16
        )
        cat_sims.scatter_reduce_(1, idx, similarities, reduce="amax", include_self=True)

        best_idx = cat_sims.argmax(dim=1)           # (N,)
        scores = cat_sims[range(N), best_idx]       # (N,)
        cat_ids = self.unique_cats[best_idx.cpu()]  # (N,)

        return [
            {"category_id": int(cat_ids[i]), "score": float(scores[i])}
            for i in range(N)
        ]

    def classify_with_fallback(
        self,
        crops: list[Image.Image],
        confidence_threshold: float = 0.3,
        fallback_category_id: int = 0,
        batch_size: int = 64,
    ) -> list[dict]:
        """Classify with fallback for low-confidence predictions.

        If max cosine similarity is below threshold, fall back to a default category.
        """
        results = self.classify(crops, batch_size)
        for r in results:
            if r["score"] < confidence_threshold:
                r["category_id"] = fallback_category_id
        return results
