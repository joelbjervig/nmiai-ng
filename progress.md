# Progress Log

## Session: 2026-03-21

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-03-21
- Actions taken:
  - Fetched all 4 competition docs, reviewed 15 training runs
  - Identified classification (0.2 mAP) as bottleneck — kNN too weak for 356 classes

### Phase 2: Supervised DINOv2 Classifier Head
- **Status:** complete
- **Started:** 2026-03-21
- Actions taken:
  - Restructured repo: src/ → train/, slurm_scripts/ → scripts/
  - Replaced ArcFace+kNN with Linear(768, 356) + CrossEntropyLoss
  - Added label smoothing (0.1), dropout (0.1), LoRA adapters (r=16)
  - Changed img_size 336 → 518, added export_checkpoint.py with LoRA merge
  - Created run_pipeline.sh orchestrator
- Training results (LoRA, 20 epochs):
  - Best val accuracy: 0.92 (epoch ~16)
  - Train/val gap: 0.07 (much tighter than classic fine-tuning's 0.12)

### Phase 3 & 4: TTA + WBF
- **Status:** complete (code)
- Actions taken:
  - Added classify_tta() with horizontal flip logit averaging
  - Added multi-scale YOLO (1280+1024) + flip + WBF in run.py
  - Added --no-tta flag

### Phase 5: Submissions
- **Status:** iterating
- Submission 1 (supervised head, 4 blocks unfrozen): local 0.79 → competition 0.63
- Submission 2 (LoRA + label smoothing): local 0.82 → competition 0.67
- Key insight: ~0.15 local→competition gap persists — distribution shift problem

## Session: 2026-03-22

### Phase 6: YOLO26 Upgrade
- **Status:** in_progress
- **Started:** 2026-03-22
- Actions taken:
  - Rewrote run.py to use onnxruntime (no ultralytics dependency at inference)
  - Added letterbox preprocessing, NMS, YOLODetector class for ONNX
  - Updated train_yolo.py: yolo26l.pt default, auto ONNX export, "robust" augmentation
  - Updated package.sh: best.onnx instead of best.pt
  - Updated pyproject.toml: ultralytics 8.1.0 → 8.4.0 for YOLO26
- Blocking: need ultralytics>=8.4 on HPC to download/train YOLO26
- Files modified:
  - run.py (rewritten — ONNX inference)
  - train/train_yolo.py (YOLO26, robust aug, ONNX export)
  - scripts/package.sh (best.onnx)
  - scripts/predict_val.slurm (best.onnx)
  - scripts/train_yolo.slurm (yolo26l defaults)
  - pyproject.toml (ultralytics==8.4.0)

## Test Results

| Test | det mAP | cls mAP | Local Score | Competition Score |
|------|---------|---------|-------------|-------------------|
| Baseline kNN | 0.90 | 0.20 | 0.69 | 0.69 |
| CE + 4 blocks | 0.84 | 0.68 | 0.79 | 0.63 |
| LoRA + smoothing | 0.84 | 0.77 | 0.82 | 0.67 |

## Error Log

| Timestamp | Error | Resolution |
|-----------|-------|------------|
| 2026-03-21 | SLURM afterok:id1:afterok:id2 | Fixed to afterok:id1:id2 |
| 2026-03-21 | ensemble_boxes ModuleNotFoundError | Added to pyproject.toml |
| 2026-03-22 | yolo26l.pt FileNotFoundError | ultralytics too old, need >=8.4 |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 6: YOLO26 upgrade, blocked on ultralytics version on HPC |
| Where am I going? | Train YOLO26 → evaluate → submit |
| What's the goal? | Close the 0.15 local→competition gap, maximize competition score |
| What have I learned? | LoRA helps (0.07 gap vs 0.12), but distribution shift is the main issue |
| What have I done? | Full ONNX inference pipeline, LoRA DINOv2, TTA+WBF, YOLO26 support |

---
*Update after completing each phase or encountering errors*
