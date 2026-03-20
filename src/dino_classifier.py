"""DINOv2 embedding-based product classifier for inference.

Loads the FP16 DINOv2 model and pre-computed reference embeddings, then
classifies cropped product images via cosine similarity lookup.

Designed to work within the competition sandbox (no os module, uses pathlib).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import timm


class DINOClassifier:
    """Two-step classifier: embed crop → nearest-neighbour in reference table."""

    def __init__(
        self,
        model_path: Path,
        embeddings_path: Path,
        model_name: str = "vit_base_patch14_dinov2",
        img_size: int = 518,
        device: str = "cuda",
        top_k: int = 1,
    ):
        self.device = device
        self.img_size = img_size
        self.top_k = top_k

        # Load DINOv2 model from FP16 weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        # Convert FP16 state dict back to FP32 for inference
        state_dict = {k: v.float() for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval().half()  # Run in FP16

        self.embed_dim = self.model.num_features

        # Load reference embeddings
        data = np.load(embeddings_path)
        self.ref_embeddings = torch.from_numpy(
            data["embedding_matrix"].astype(np.float32)
        ).half().to(device)  # (N_cats, D)
        self.category_ids = data["category_ids"]  # (N_cats,) int32

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"DINOClassifier ready: {model_name}, {len(self.category_ids)} ref categories, device={device}")

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

        Returns list of dicts with keys:
            - category_id: int (COCO category ID, 0-355)
            - score: float (cosine similarity, 0-1)
        """
        if not crops:
            return []

        embeddings = self.embed_crops(crops, batch_size)  # (N, D)

        # Cosine similarity against all reference embeddings
        # ref_embeddings is already L2-normalised, so dot product = cosine sim
        similarities = embeddings @ self.ref_embeddings.T  # (N, N_cats)

        results = []
        for i in range(len(crops)):
            sims = similarities[i]
            top_idx = sims.argmax().item()
            score = sims[top_idx].item()
            cat_id = int(self.category_ids[top_idx])
            results.append({"category_id": cat_id, "score": float(score)})

        return results

    def classify_with_fallback(
        self,
        crops: list[Image.Image],
        confidence_threshold: float = 0.3,
        fallback_category_id: int = 0,
        batch_size: int = 64,
    ) -> list[dict]:
        """Classify with fallback for low-confidence predictions.

        If cosine similarity is below threshold, fall back to a default category.
        This is useful for the ~5% of annotations without reference images.
        """
        results = self.classify(crops, batch_size)
        for r in results:
            if r["score"] < confidence_threshold:
                r["category_id"] = fallback_category_id
        return results
