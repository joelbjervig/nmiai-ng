# Findings & Decisions

## Requirements
- NM i AI 2026: NorgesGruppen Data — Object Detection on grocery store shelves
- Detect and classify grocery products using bounding boxes
- Submit `run.py` in a `.zip` file, executed in sandboxed Docker with NVIDIA L4 GPU
- Score = 0.7 × detection_mAP + 0.3 × classification_mAP (IoU ≥ 0.5)

---

## Competition Overview (source: /docs/norgesgruppen-data/overview)

**NM i AI 2026** — Norwegian AI Championship, March 19-22, 2026 (69 hours). 1,000,000 NOK prize pool. Three independent tasks; this project focuses on **NorgesGruppen Data: Object Detection**.

### Objective
Detect and classify grocery products on store shelves.

### Workflow
1. Download training data from competition platform
2. Train an object detection model locally
3. Create a `run.py` script that processes shelf images
4. Submit code as `.zip` file
5. System executes code in sandboxed Docker with NVIDIA L4 GPU (24GB VRAM)

### Training Dataset
- **COCO Dataset** (~864 MB): 248 shelf images, ~22,700 bounding box annotations
- **Product Reference Images** (~60 MB): 327 products with multi-angle photos (main, front, back, left, right, top, bottom)
- **356 product categories** (IDs 0-355)
- Four store sections: Egg, Frokost, Knekkebrod, Varmedrikker

### Annotation Format (COCO)
```
bbox: [x, y, width, height] in pixels
product_code: barcode identifier
corrected: boolean for manually verified annotations
```

### Sandbox Specifications
| Resource | Specification |
|----------|---------------|
| Python | 3.11 |
| CPU | 4 vCPU |
| RAM | 8 GB |
| GPU | NVIDIA L4 (24 GB VRAM) |
| CUDA | 12.4 |
| Network | None (offline) |
| Timeout | 300 seconds |

### Pre-installed Packages
PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, opencv-python-headless, albumentations, Pillow, numpy, scipy, scikit-learn, pycocotools, ensemble-boxes, timm, supervision, safetensors.

**Note:** Runtime `pip install` is blocked.

### Submission Limits
- Concurrent submissions: 2 per team
- Daily submission quota: 3 per team
- Infrastructure error allowance: 2 daily (exempt from quota)
- Resets at midnight UTC

---

## Submission Format (source: /docs/norgesgruppen-data/submission)

### Zip Structure
```
submission.zip
├── run.py          # Required entry point (MUST be at root)
├── model.onnx      # Optional: model weights
└── utils.py        # Optional: helper code
```

### File Limits
| Constraint | Limit |
|-----------|-------|
| Uncompressed size | 420 MB |
| Total files | 1000 |
| Python files | 10 |
| Weight files | 3 |
| Total weight size | 420 MB |
| Allowed types | .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy |

### run.py Contract
```bash
python run.py --input /data/images --output /output/predictions.json
```

**Input:** `/data/images/` contains JPEG shelf images: `img_XXXXX.jpg`

**Output:** JSON array:
```json
[
  {
    "image_id": 42,
    "category_id": 0,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | int | Numeric ID extracted from filename |
| `category_id` | int | Product category (0-355) from annotations.json |
| `bbox` | [x, y, w, h] | Bounding box in COCO format (pixels) |
| `score` | float | Confidence score (0-1) |

### Security Restrictions (Blocked Imports)
- System: `os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`
- Serialization: `pickle`, `marshal`, `shelve`, `shutil`
- Config: `yaml`
- Network: `requests`, `urllib`, `http.client`
- Threading: `multiprocessing`, `threading`, `signal`, `gc`
- Code execution: `eval()`, `exec()`, `compile()`, `__import__()`, dangerous `getattr()`

Use `pathlib` for file operations and `json` for configuration.

### Creating Zip (macOS)
```bash
cd my_submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```

### Supported Frameworks
| Framework | Models | Version |
|-----------|--------|---------|
| ultralytics 8.1.0 | YOLOv8n/s/m/l/x, YOLOv5u, RT-DETR | `ultralytics==8.1.0` |
| torchvision 0.21.0 | Faster R-CNN, RetinaNet, SSD, FCOS, Mask R-CNN | `torchvision==0.21.0` |
| timm 0.9.12 | ResNet, EfficientNet, ViT, Swin, ConvNeXt | `timm==0.9.12` |

**Unsupported** (need ONNX export or custom code): YOLOv9, YOLOv10, YOLO11, RF-DETR, Detectron2, MMDetection, HuggingFace Transformers.

---

## Scoring Details (source: /docs/norgesgruppen-data/scoring)

### Hybrid Scoring Formula
```
Score = 0.7 × detection_mAP + 0.3 × classification_mAP
```
Both use mAP@0.5 (IoU threshold 0.5).

### Detection Component (70%)
- Predictions matched to closest ground truth bbox
- Match = IoU ≥ 0.5 (category ignored)
- Rewards precise spatial positioning

### Classification Component (30%)
- Requires IoU ≥ 0.5 AND matching `category_id`
- 356 product categories (IDs 0-355)

### Detection-Only Pathway
- Submit all `category_id: 0` → max score 0.70 (70%)
- Proper classification unlocks remaining 30%

### Leaderboard
- Public leaderboard: public test set scores
- Final ranking: hidden private test set
- Teams can manually select any completed submission for final eval ("Select for final")

---

## Examples & Tips (source: /docs/norgesgruppen-data/examples)

### Random Baseline (for verification)
```python
import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions = []
    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        for _ in range(random.randint(5, 20)):
            predictions.append({
                "image_id": image_id,
                "category_id": random.randint(0, 356),
                "bbox": [
                    round(random.uniform(0, 1500), 1),
                    round(random.uniform(0, 800), 1),
                    round(random.uniform(20, 200), 1),
                    round(random.uniform(20, 200), 1),
                ],
                "score": round(random.uniform(0.01, 1.0), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

### YOLOv8 Example
```python
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    predictions = []

    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        results = model(str(img), device=device, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [round(x1, 1), round(y1, 1),
                            round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[i].item()), 3),
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

**Caveat:** Pretrained COCO model outputs COCO class IDs (0-79), not product IDs (0-355). Must fine-tune with `nc=357`.

### ONNX Export
```python
# From ultralytics
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=17)

# From generic PyTorch
import torch
dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy, "model.onnx", opset_version=17)
```

### ONNX Inference
```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: arr})
```

### Common Errors & Fixes
| Error | Solution |
|-------|----------|
| `run.py not found at zip root` | Zip contents directly, not the folder |
| `Disallowed file type: __MACOSX/...` | Use: `zip -r ../sub.zip . -x ".*" "__MACOSX/*"` |
| `Disallowed file type: .bin` | Rename `.bin` → `.pt` or convert to `.safetensors` |
| `Security scan found violations` | Remove subprocess, socket, os imports; use pathlib |
| `No predictions.json in output` | Verify run.py writes to `--output` path |
| `Timed out after 300s` | Ensure GPU usage; consider smaller models |
| `Exit code 137` | OOM (8 GB RAM); reduce batch size or use FP16 |
| `Exit code 139` | Segfault; weight version mismatch; re-export or use ONNX |
| `ModuleNotFoundError` | Export to ONNX or include model code inline |
| `KeyError / RuntimeError on model load` | Version mismatch; pin exact versions or use ONNX |

### Tips
- Start with the random baseline to verify setup
- GPU is available — larger models (YOLOv8m/l/x) feasible within 300s
- Use `torch.cuda.is_available()` for adaptive code
- FP16 quantization recommended (smaller weights, faster inference)
- ONNX with `CUDAExecutionProvider` works for any framework
- Process images one at a time; use `torch.no_grad()` during inference
- Match only packages you actually use in training

---

## Research Findings
- 248 training images with ~22,700 annotations = ~91 objects per image (dense shelves)
- 327 products with reference images available for potential few-shot/metric learning
- Detection-only baseline caps at 70% — classification is essential for top scores
- L4 GPU with 24GB VRAM allows large models (YOLOv8l/x)
- 420 MB weight limit accommodates most single models
- kNN classification with DINOv2 embeddings only achieves 0.2 cls mAP — too noisy for 356 similar classes
- Supervised Linear(768, 356) head on DINOv2 is the standard transfer learning approach and should dramatically improve classification
- CrossEntropyLoss preferred over ArcFace for closed-set classification (simpler, fewer hyperparams)
- Training DINOv2 at 518px (native patch-14 resolution) avoids mismatch with inference resolution
- A30 24GB GPU on HPC fits DINOv2 518px training with batch=32 + grad checkpointing

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Two-stage YOLO (detect) + DINOv2 (classify) | YOLO strong at detection (0.9 mAP), weak at fine-grained 356-class classification |
| Supervised linear head over kNN | kNN gives 0.2 cls mAP; linear head learns decision boundaries between confusable classes |
| CrossEntropyLoss over ArcFace | Closed-set, simpler, directly optimises classification |
| Fine-tune DINOv2 at 518px | Match inference resolution, avoid representation mismatch |
| Batch size 32 for DINOv2 training | Safe for 24GB A30 at 518px with grad checkpointing |

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
- Overview: https://app.ainm.no/docs/norgesgruppen-data/overview
- Submission: https://app.ainm.no/docs/norgesgruppen-data/submission
- Scoring: https://app.ainm.no/docs/norgesgruppen-data/scoring
- Examples: https://app.ainm.no/docs/norgesgruppen-data/examples
- MCP Server: `claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp`

## Visual/Browser Findings
- All four documentation pages fetched and captured on 2026-03-21

---
*Update this file after every 2 view/browser/search operations*
