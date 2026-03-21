"""DINOv2 supervised product classifier for inference.

Loads the fine-tuned DINOv2 backbone (FP16) and a trained linear classification
head, then classifies cropped product images via a forward pass.

Supports test-time augmentation (TTA) by averaging logits across original
and horizontally flipped crops for more robust classification.

Crops are padded to square before resizing to preserve aspect ratio.

Designed to work within the competition sandbox (no os module, uses pathlib).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    """DINOv2 backbone + linear classification head.

    Loads fine-tuned backbone weights and a trained Linear(768, num_classes)
    head exported by train/train_dino.py.
    """

    def __init__(
        self,
        model_path: Path,
        head_path: Path,
        model_name: str = "vit_base_patch14_dinov2",
        img_size: int = 518,
        device: str = "cuda",
    ):
        self.device = device
        self.img_size = img_size

        # Load DINOv2 backbone from FP16 weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = {k: v.float() for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).eval().half()

        embed_dim = self.model.num_features

        # Load trained classification head
        head_data = np.load(head_path, allow_pickle=True).item()
        weight = torch.from_numpy(head_data["weight"].astype(np.float32))
        bias = torch.from_numpy(head_data["bias"].astype(np.float32))
        self.label_to_catid = {int(k): int(v) for k, v in head_data["label_to_catid"].items()}

        num_classes = weight.shape[0]
        self.head = nn.Linear(embed_dim, num_classes)
        self.head.weight.data = weight
        self.head.bias.data = bias
        self.head = self.head.to(device).eval().half()

        # Aspect-ratio-preserving preprocessing: pad to square, then resize
        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"DINOClassifier ready: {model_name}, {num_classes} classes, device={device}")

    @torch.no_grad()
    def _get_logits(self, crops: list[Image.Image], batch_size: int = 64) -> torch.Tensor:
        """Get raw logits for a list of crops."""
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
        """Convert logits tensor to list of {category_id, score} dicts."""
        probs = logits.softmax(dim=-1)
        scores, labels = probs.max(dim=-1)
        return [
            {"category_id": self.label_to_catid[label.item()], "score": float(score)}
            for score, label in zip(scores.cpu(), labels.cpu())
        ]

    @torch.no_grad()
    def classify(self, crops: list[Image.Image], batch_size: int = 64) -> list[dict]:
        """Classify cropped product images.

        Returns list of dicts with keys:
            - category_id: int (COCO category ID, 0-355)
            - score: float (softmax confidence, 0-1)
        """
        if not crops:
            return []
        logits = self._get_logits(crops, batch_size)
        return self._logits_to_results(logits)

    @torch.no_grad()
    def classify_tta(self, crops: list[Image.Image], batch_size: int = 64) -> list[dict]:
        """Classify with test-time augmentation (horizontal flip).

        Averages logits from original + flipped crops before softmax.
        More robust than single-pass classification.
        """
        if not crops:
            return []

        # Original
        logits_orig = self._get_logits(crops, batch_size)

        # Horizontal flip
        flipped = [c.transpose(Image.FLIP_LEFT_RIGHT) for c in crops]
        logits_flip = self._get_logits(flipped, batch_size)

        # Average logits (more stable than averaging probabilities)
        logits_avg = (logits_orig + logits_flip) / 2.0
        return self._logits_to_results(logits_avg)
