# Task Plan: NorgesGruppen Object Detection (NM i AI 2026)

## Goal
Maximize competition score (Score = 0.7 × detection_mAP + 0.3 × classification_mAP) on unseen test set. Focus on closing the local→competition generalization gap.

## Current Phase
Phase 6 (YOLO26 upgrade — training in progress)

## Phases

### Phase 1: Requirements & Discovery
- [x] All tasks complete
- **Status:** complete

### Phase 2: Supervised DINOv2 Classifier Head
- [x] Linear(768, 356) + CrossEntropyLoss (replaced ArcFace + kNN)
- [x] Fine-tune at 518px, export backbone FP16 + cls_head.npy
- [x] Label smoothing (0.1) + dropout (0.1) for regularization
- [x] LoRA adapters (r=16, alpha=32) — better generalization than block unfreezing
- **Status:** complete
- **Results:** local val acc 0.92 (LoRA), local cls mAP 0.77, competition cls mAP ~0.25

### Phase 3: Crop TTA
- [x] classify_tta() with horizontal flip logit averaging
- **Status:** complete

### Phase 4: Detection TTA + WBF
- [x] Multi-scale YOLO (1280 + 1024) + flip + ensemble-boxes WBF
- [x] --no-tta flag for fast fallback
- **Status:** complete

### Phase 5: Package & Submit
- [x] ONNX-based submission pipeline (no ultralytics at inference)
- [x] export_checkpoint.py with LoRA merge support
- [x] Submitted: 0.79 local → 0.63 competition, 0.82 local → 0.67 competition
- **Status:** complete (iterating)

### Phase 6: YOLO26 Upgrade
- [x] Rewrite run.py to use onnxruntime (no ultralytics dependency)
- [x] train_yolo.py: YOLO26 support + auto ONNX export + "robust" augmentation
- [x] package.sh: best.onnx instead of best.pt
- [ ] Install ultralytics>=8.4.0 on HPC for YOLO26
- [ ] Train YOLO26l with robust augmentation + freeze=16
- [ ] Evaluate YOLO26 vs YOLOv8 on competition
- **Status:** in_progress

## Key Questions
1. ~~ArcFace vs CE?~~ Resolved: CE
2. ~~How many blocks to unfreeze?~~ Resolved: LoRA (0 blocks, adapters instead)
3. Why is local→competition gap so large (0.15)? Distribution shift in test set.
4. Does YOLO26 improve detection on the competition test set?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Two-stage YOLO + DINOv2 | YOLO for detection, DINOv2 for fine-grained classification |
| Supervised linear head over kNN | kNN gives 0.2 cls mAP, linear head gives 0.77 locally |
| CrossEntropyLoss + label smoothing 0.1 | Closed-set, regularized |
| LoRA (r=16) over block unfreezing | Better generalization: 0.92 val acc vs 0.82, tighter train/val gap |
| ONNX inference (no ultralytics in submission) | Sandbox has ultralytics 8.1.0 which doesn't support YOLO26 |
| YOLO26l over YOLOv8l | Smaller model (~51 MB vs 83 MB), better small object detection (ProgLoss + STAL) |
| Robust augmentation for YOLO | Stronger color/geometric aug to combat distribution shift |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| SLURM dependency format | afterok:id1:afterok:id2 invalid | Fixed to afterok:id1:id2 |
| ensemble_boxes not installed on HPC | ModuleNotFoundError | Added to pyproject.toml |
| Local→competition gap 0.15 | Classic fine-tuning overfits | Added LoRA, label smoothing, dropout |
| ultralytics 8.1.0 can't load YOLO26 | FileNotFoundError: yolo26l.pt | Need ultralytics>=8.4.0 on HPC |

## Competition Score History
| Submission | Local Score | Competition Score | Notes |
|------------|-------------|-------------------|-------|
| Baseline (kNN) | 0.69 | 0.69 | det=0.9, cls=0.2 |
| Supervised head (CE, 4 blocks) | 0.79 | 0.63 | Big generalization gap |
| LoRA + label smoothing | 0.82 | 0.67 | Gap slightly smaller |

## Notes
- Competition ends March 22, 2026 15:00 CET
- Daily submission quota: 3 (resets midnight UTC)
- Class imbalance: 74 classes with <5 samples, median=28, max=422
- 420 MB weight limit: YOLO26l ONNX ~51 MB + DINOv2 FP16 ~170 MB + cls_head ~1 MB = ~222 MB (comfortable)
