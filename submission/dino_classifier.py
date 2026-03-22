"""DINOv2 product classifier for inference.

Supports two classification modes:
  1. Linear head: forward pass through trained Linear(768, 356) head
  2. kNN: cosine similarity against mean reference embeddings per class

Both modes support TTA (horizontal flip averaging).

Designed to work within the competition sandbox (no os module, uses pathlib).
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
    """DINOv2 backbone with linear head or kNN classification.

    Mode is auto-detected based on which files exist:
      - head_path (cls_head.npy) → linear head mode
      - embeddings_path (ref_embeddings.npy) → kNN mode
      - Both → linear head (default), switchable via use_knn flag

    Optional: extract_layer selects which transformer block to use for embeddings.
      - None (default): use final CLS token (standard)
      - 0-11: use CLS token from that intermediate block (ViT-B has 12 blocks)
    """

    def __init__(
        self,
        model_path: Path,
        head_path: Path = None,
        embeddings_path: Path = None,
        model_name: str = "vit_base_patch14_dinov2",
        img_size: int = 518,
        device: str = "cuda",
        use_knn: bool = False,
        extract_layer: int = None,
    ):
        self.device = device
        self.img_size = img_size
        self.use_knn = use_knn
        self.extract_layer = extract_layer
        self._intermediate_output = None

        # Load DINOv2 backbone from FP16 weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = {k: v.float() for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval().half()

        embed_dim = self.model.num_features

        # Register hook for intermediate layer extraction
        if extract_layer is not None:
            total_blocks = len(self.model.blocks)
            if extract_layer < 0 or extract_layer >= total_blocks:
                raise ValueError(f"extract_layer={extract_layer} out of range [0, {total_blocks-1}]")

            def hook_fn(module, input, output):
                # ViT block output: (B, N_tokens, D) — take CLS token (index 0)
                self._intermediate_output = output[:, 0, :]

            self.model.blocks[extract_layer].register_forward_hook(hook_fn)
            print(f"  Extracting features from block {extract_layer}/{total_blocks-1}")
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
            print(f"  Linear head loaded: {num_classes} classes")

        # Load kNN reference embeddings
        if embeddings_path is not None and embeddings_path.exists():
            ref = np.load(embeddings_path).astype(np.float32)
            self.ref_embeddings = torch.from_numpy(ref).half().to(device)
            catids_path = embeddings_path.with_name("ref_catids.json")
            with open(catids_path) as f:
                raw = json.load(f)
            self.knn_label_to_catid = {int(k): int(v) for k, v in raw.items()}
            print(f"  kNN embeddings loaded: {ref.shape[0]} classes, dim={ref.shape[1]}")

        # Determine mode
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

        # Preprocessing
        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"DINOClassifier ready: {model_name}, mode={self._mode}, device={device}")

    @torch.no_grad()
    def _embed(self, crops: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Embed crops → (N, D) L2-normalized FP16 tensor."""
        all_embs = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            tensors = [self.transform(c.convert("RGB")) for c in batch_crops]
            batch = torch.stack(tensors).to(self.device).half()
            output = self.model(batch)  # also triggers hook if extract_layer is set
            features = self._intermediate_output if self.extract_layer is not None else output
            features = F.normalize(features, p=2, dim=-1)
            all_embs.append(features)
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def _get_logits(self, crops: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Get raw logits (linear head) or cosine similarities (kNN)."""
        if self._mode == "knn":
            embeddings = self._embed(crops, batch_size)
            # Cosine similarity against reference embeddings (already L2-normed)
            return embeddings @ self.ref_embeddings.T  # (N, num_classes)
        else:
            all_logits = []
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i:i + batch_size]
                tensors = [self.transform(c.convert("RGB")) for c in batch_crops]
                batch = torch.stack(tensors).to(self.device).half()
                features = self.model(batch)
                logits = self.head(features)
                all_logits.append(logits)
            return torch.cat(all_logits, dim=0)

    def _logits_to_results(self, logits: torch.Tensor) -> list[dict]:
        """Convert logits/similarities to list of {category_id, score} dicts."""
        if self._mode == "knn":
            # For kNN, scores are cosine similarities (0-1 range)
            scores, labels = logits.max(dim=-1)
            label_map = self.knn_label_to_catid
        else:
            probs = logits.softmax(dim=-1)
            scores, labels = probs.max(dim=-1)
            label_map = self.label_to_catid
        return [
            {"category_id": label_map[label.item()], "score": float(score)}
            for score, label in zip(scores.cpu(), labels.cpu())
        ]

    @torch.no_grad()
    def classify(self, crops: list[Image.Image], batch_size: int = 64) -> list[dict]:
        """Classify cropped product images."""
        if not crops:
            return []
        logits = self._get_logits(crops, batch_size)
        return self._logits_to_results(logits)

    @torch.no_grad()
    def classify_tta(self, crops: list[Image.Image], batch_size: int = 64) -> list[dict]:
        """Classify with TTA (horizontal flip, average logits/similarities)."""
        if not crops:
            return []
        logits_orig = self._get_logits(crops, batch_size)
        flipped = [c.transpose(Image.FLIP_LEFT_RIGHT) for c in crops]
        logits_flip = self._get_logits(flipped, batch_size)
        logits_avg = (logits_orig + logits_flip) / 2.0
        return self._logits_to_results(logits_avg)
