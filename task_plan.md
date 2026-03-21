# Task Plan: NorgesGruppen Object Detection (NM i AI 2026)

## Goal
Maximize competition score (Score = 0.7 × detection_mAP + 0.3 × classification_mAP) by fixing classification (currently 0.2) with supervised DINOv2 classifier head, then improving detection with TTA + ensemble.

## Current Phase
Phase 1

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand competition rules, scoring, submission format
- [x] Document findings from all 4 competition docs
- [x] Review current codebase and training runs
- [x] Assess current approaches and identify bottlenecks
- **Status:** complete

### Phase 2: Supervised DINOv2 Classifier Head
- [x] Modify `train_dino.py`: Linear(768, 356) head + CrossEntropyLoss (replaced ArcFace)
- [x] Fine-tune at image size 518 (match inference resolution)
- [x] Export backbone (FP16) and classifier head weights (cls_head.npy) separately
- [x] Update slurm script defaults (img_size=518, batch=32, no ArcFace args)
- [x] Update package.sh for cls_head.npy instead of ref_embeddings
- [x] Rewrite `dino_classifier.py`: linear head forward pass (removed kNN, ref_embeddings, fallback)
- [x] Rewrite `run.py`: loads cls_head.npy, no more ref_embeddings or confidence fallback
- [x] Full dead-code sweep: removed all ArcFace, kNN, ref_embeddings references across codebase
- [ ] Try label smoothing in CrossEntropyLoss for regularization (e.g. label_smoothing=0.1)
- **Status:** in_progress

### Phase 3: Crop Test-Time Augmentation (TTA)
- [ ] Add TTA to `dino_classifier.py` (horizontal flip, scale jitter, color jitter)
- [ ] For each crop: create N augmented versions, embed all, average embeddings, then classify
- [ ] Benchmark TTA impact on classification mAP locally with `eval_val.py`
- **Status:** pending

### Phase 4: Detection TTA + Weighted Boxes Fusion (WBF)
- [ ] Add multi-scale inference to YOLO detector (e.g., 1280 + 1024, horizontal flip)
- [ ] Use `ensemble-boxes` (pre-installed) for Weighted Boxes Fusion to merge predictions
- [ ] Tune WBF parameters (iou_thr, skip_box_thr, weights)
- [ ] Benchmark detection mAP improvement locally
- **Status:** pending

### Phase 5: Package & Submit
- [x] Update `package.sh` for new submission structure (backbone + cls_head.npy)
- [ ] Verify submission.zip ≤ 420 MB
- [ ] Verify run.py works with sandbox constraints (blocked imports, 300s timeout)
- [ ] Submit and evaluate on competition leaderboard
- **Status:** pending

## Key Questions
1. ~~ArcFace vs cross-entropy?~~ Resolved: using CrossEntropyLoss (simpler, closed-set)
2. How many DINOv2 blocks to unfreeze? (Currently 4, may need to tune)
3. Does TTA on crops fit within 300s timeout with ~90+ detections per image?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Two-stage YOLO + DINOv2 over end-to-end multi-class YOLO | YOLO detection is strong (0.9 mAP); classification needs specialized model. 356 classes with ~64 samples/class too thin for YOLO's conv head |
| Supervised linear head over nearest-neighbor lookup | kNN gives 0.2 cls mAP — linear head learns decision boundaries between confusable classes, much more discriminative |
| Fine-tune DINOv2 at 518px | Must match inference resolution. Previous 336px fine-tuning → 518px inference creates representation mismatch |
| Drop reference embeddings pipeline | Supervised head replaces kNN entirely. Smaller submission, simpler pipeline |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

## Notes
- Current score: 0.7 × 0.9 + 0.3 × 0.2 = 0.69
- Target: 0.85-0.90 (0.95 det × 0.75 cls)
- Priority order: Phase 2 (biggest impact) → Phase 3 → Phase 4
- YOLO+DINOv2 ensemble voting with multi-class YOLO deferred — revisit only if time permits
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions
- Log ALL errors - they help avoid repetition
