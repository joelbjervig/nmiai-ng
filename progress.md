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
- **Status:** in_progress — training done, evaluating
- **Started:** 2026-03-22
- Actions taken:
  - Rewrote run.py to use onnxruntime (no ultralytics dependency at inference)
  - Added letterbox preprocessing, NMS, YOLODetector class for ONNX
  - Handles both YOLOv8 [1,5,N] and YOLO26 e2e [1,300,6] output formats
  - Updated train_yolo.py: yolo26l.pt default, auto ONNX export, "robust" augmentation
  - Updated package.sh: best.onnx instead of best.pt
  - Updated pyproject.toml: ultralytics 8.1.0 → 8.4.0 for YOLO26
  - Fixed 5 failed YOLO26 training attempts before getting stable training (detect5-8)
  - Best run: yolo26l_detect8 (freeze=16, batch=8, lr=0.0005, optimizer=auto, 500 epochs)
  - Created export_yolo_onnx.slurm for manual ONNX export
  - Fixed bug: detection flip TTA was dead code (required multi-scale)
  - Fixed bug: YOLO26 e2e output format different from YOLOv8
  - Added det×cls confidence score multiplication for mAP ranking

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
| 2026-03-22 | ONNX fixed input size 1280 | Disabled multi-scale TTA, use flip only |
| 2026-03-22 | YOLO26 e2e output [1,300,6] ≠ YOLOv8 [1,5,N] | Added format detection in postprocess |
| 2026-03-22 | Detection flip TTA was dead code | Fixed: trigger on DETECT_FLIP not just multi-scale |
| 2026-03-22 | 0 predictions from wrong ONNX (broken detect run) | Used detect8 weights instead |

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Phase 6: YOLO26 trained (detect8), evaluating with ONNX pipeline |
| Where am I going? | Eval YOLO26 locally → submit → compare with YOLOv8 baseline |
| What's the goal? | Close the 0.15 local→competition gap, maximize competition score |
| What have I learned? | LoRA helps (0.07 gap vs 0.12), YOLO26 e2e has different output format, distribution shift is main issue |
| What have I done? | Full ONNX pipeline, LoRA DINOv2, TTA+WBF, YOLO26 training + export, pipeline audit |

---
*Update after completing each phase or encountering errors*
