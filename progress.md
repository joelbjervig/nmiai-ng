# Progress Log

## Session: 2026-03-21

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-03-21
- Actions taken:
  - Fetched all 4 competition docs (overview, submission, scoring, examples)
  - Captured full competition details into findings.md
  - Reviewed all training runs (15 runs), hyperparameters, and metrics
  - Assessed both approaches (YOLO-only vs YOLO+DINOv2)
  - Identified classification (0.2 mAP) as the bottleneck — kNN too weak for 356 classes
- Files created/modified:
  - task_plan.md (created)
  - findings.md (created + populated)
  - progress.md (created)

### Phase 2: Supervised DINOv2 Classifier Head
- **Status:** complete (code — awaiting HPC training)
- **Started:** 2026-03-21
- Actions taken:
  - Restructured repo: src/ → train/, slurm_scripts/ → scripts/
  - Removed dead approaches: run_yolo_cls.py, build_embeddings, build_irl_embeddings, multiclass slurm
  - Rewrote train/train_dino.py: replaced ArcFace with Linear(768, 356) + CrossEntropyLoss
  - Changed img_size default from 336 → 518 (match inference resolution)
  - Rewrote dino_classifier.py: linear head forward pass replaces kNN/ref_embeddings
  - Rewrote run.py: loads cls_head.npy, removed fallback/confidence threshold
  - Updated package.sh: packages cls_head.npy instead of ref_embeddings.npy
  - Created scripts/run_pipeline.sh: full SLURM orchestrator with dependency chains
  - Aligned all defaults across slurm scripts and Python scripts
  - Full dead-code sweep across entire codebase
  - Removed plot_umap.py (unused)
- Files created/modified:
  - train/train_dino.py (rewritten)
  - dino_classifier.py (rewritten)
  - run.py (rewritten)
  - scripts/run_pipeline.sh (new)
  - scripts/package.sh (updated)
  - scripts/package.slurm (updated)
  - scripts/train_dino.slurm (updated)
  - scripts/train_yolo.slurm (updated)
  - scripts/prepare_data.slurm (updated)
  - train/train_yolo.py (defaults fixed)
  - .gitignore (updated)
- Files removed:
  - run_yolo_cls.py, package_yolo_cls.sh
  - src/build_embeddings.py, src/build_irl_embeddings.py
  - train/plot_umap.py, scripts/plot_umap.slurm
  - 5 obsolete slurm scripts
  - submission/ dir, stale zips

### Next: Run pipeline on HPC
- `git pull && ./scripts/run_pipeline.sh --skip-prepare`
- Monitor: `squeue -u $USER` and `tail -f output/nmiai-*.out`
- After training: check eval scores, then proceed to Phase 3 (crop TTA)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Baseline (pre-refactor) | val set | det=0.9, cls=0.2 | det=0.9, cls=0.2 | baseline |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 2 complete (code), awaiting HPC training |
| Where am I going? | Train on HPC → Phase 3 (crop TTA) → Phase 4 (detection TTA) |
| What's the goal? | Maximize Score = 0.7 × det_mAP + 0.3 × cls_mAP (target: 0.85-0.90) |
| What have I learned? | See findings.md — kNN fails at 356 classes, supervised head is the fix |
| What have I done? | Full codebase rewrite: supervised head, repo restructure, pipeline orchestrator |

---
*Update after completing each phase or encountering errors*
