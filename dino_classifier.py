"""DINOv2 product classifier for inference.

Supports two modes:
  1. Linear head (default): forward pass through trained Linear(768, 356)
  2. kNN: cosine similarity against mean reference embeddings per class

Designed for competition sandbox (no os module, uses pathlib).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import timm


class PadToSquare:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        max_side = max(w, h)
        padded = Image.new("RGB", (max_side, max_side), (114, 114, 114))
        padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return padded


class DINOClassifier:
    def __init__(
        self,
        model_path: Path,
        head_path: Path = None,
        embeddings_path: Path = None,
        model_name: str = "vit_base_patch14_dinov2",
        img_size: int = 518,
        device: str = "cuda",
        use_knn: bool = False,
    ):
        self.device = device

        # Load backbone
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = {k: v.float() for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval().half()

        embed_dim = self.model.num_features
        self.head = None
        self.ref_embeddings = None
        self.label_to_catid = None

        # Load linear head
        if head_path is not None and head_path.exists():
            head_data = np.load(head_path, allow_pickle=True).item()
            weight = torch.from_numpy(head_data["weight"].astype(np.float32))
            bias = torch.from_numpy(head_data["bias"].astype(np.float32))
            self.label_to_catid = {int(k): int(v) for k, v in head_data["label_to_catid"].items()}
            num_classes = weight.shape[0]
            self.head = nn.Linear(embed_dim, num_classes)
            self.head.weight.data = weight
            self.head.bias.data = bias
            self.head = self.head.to(device).eval().half()

        # Load kNN embeddings
        if embeddings_path is not None and embeddings_path.exists():
            ref = np.load(embeddings_path).astype(np.float32)
            self.ref_embeddings = torch.from_numpy(ref).half().to(device)
            catids_path = embeddings_path.with_name("ref_catids.json")
            with open(catids_path) as f:
                raw = json.load(f)
            self.knn_label_to_catid = {int(k): int(v) for k, v in raw.items()}

        # Pick mode
        if use_knn and self.ref_embeddings is not None:
            self._mode = "knn"
            if self.label_to_catid is None:
                self.label_to_catid = self.knn_label_to_catid
        elif self.head is not None:
            self._mode = "linear"
        elif self.ref_embeddings is not None:
            self._mode = "knn"
            self.label_to_catid = self.knn_label_to_catid
        else:
            raise ValueError("Need either head_path or embeddings_path")

        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"DINOClassifier ready: {model_name}, mode={self._mode}, device={device}")

    @torch.no_grad()
    def classify(self, crops: list[Image.Image], batch_size: int = 64) -> list[dict]:
        if not crops:
            return []
        all_results = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            tensors = [self.transform(c.convert("RGB")) for c in batch_crops]
            batch = torch.stack(tensors).to(self.device).half()
            features = self.model(batch)

            if self._mode == "knn":
                features = F.normalize(features, p=2, dim=-1)
                sims = features @ self.ref_embeddings.T
                scores, labels = sims.max(dim=-1)
                label_map = self.knn_label_to_catid
            else:
                logits = self.head(features)
                probs = logits.softmax(dim=-1)
                scores, labels = probs.max(dim=-1)
                label_map = self.label_to_catid

            for score, label in zip(scores.cpu(), labels.cpu()):
                all_results.append({
                    "category_id": label_map[label.item()],
                    "score": float(score),
                })
        return all_results
