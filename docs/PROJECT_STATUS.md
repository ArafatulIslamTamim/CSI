# Project Status — What Is Completed and What Is Running

Last updated: 2026-08-24

This document is the full status of the WiFi CSI → human pose project after
moving to the **3-day dataset** (`day1` / `day2` / `day3`).

---

## 1. One-line summary

| Area | Status |
|------|--------|
| Old 7-subject pipeline outputs | Cleared; **60 checkpoints archived** |
| 3-day raw data in day folders | **Done** (~262 CSI+video pairs) |
| Pipeline code for day folders | **Done** (`src/raw_paths.py` + updates) |
| 5 FPS video + 5 CSI per frame | **Configured** |
| Frames + YOLO labels + dataset NPZ | **Done** (built for current settings) |
| Train / val / test split by person ID | **Configured** |
| Empty-room 3-class training design | **Configured** |
| Cross-day CSI analysis + docs | **Done** |
| **Current training run** | **RUNNING** (`day3_id007`, epoch ~27, best epoch 12) |
| Evaluate / visualize / live (this run) | **Not finished yet** (wait for train) |

---

## 2. What you completed (chronological)

### A. Earlier research pipeline (archived)

Completed on the previous single-folder, 7-subject dataset:

1. Full offline pipeline: CSI parse → bursts → YOLO-Pose GT → SAM → train/eval/viz  
2. Empty-room as 3rd class + live replay inference with skeleton gating  
3. Experiment stack P0–P5 in code:
   - P0 GT QA + MPJPE  
   - P1 root-relative pose  
   - P2 bone/angle losses  
   - P3 CSI augmentation + AdaBN-style adapt-norm  
   - P4 subject-adversarial (GRL)  
   - P5 GCN decoder  
4. LOSO runner + aggregate (`scripts/run_loso_experiment.py`)  
5. Measured LOSO: `baseline3c` and `rootrel` full 7 folds; `rootrel_geo` only subject 001  

**Saved (not deleted):**

```text
archive/previous_7subject_pipeline/checkpoints/   # 60 .pt files
archive/previous_7subject_pipeline/evaluation/loso_summary.json
archive/previous_7subject_pipeline/CHANGES_AND_RESULTS.md (copy)
```

Narrative of that era: `docs/CHANGES_AND_RESULTS.md`

### B. Reset for 3-day collection

1. Cleared previous **inputs** (old flat CSI/videos) and **outputs** (metrics, viz, logs)  
2. Kept archived models (above)  
3. Created day folders:

```text
data/raw/csi/day1|day2|day3/
data/raw/videos/day1|day2|day3/
```

4. Updated pipeline to discover files inside day folders (`src/raw_paths.py`)  
5. Subject ID parsed as person (`001`…`007`), not as `day1`  
6. Empty sessions auto-split by day: day1→train, day2→val, day3→test  

### C. Sync design for this collection

Configured in `src/config.py`:

| Setting | Value | Meaning |
|---------|-------|---------|
| `TARGET_VIDEO_FPS` | **5.0** | 5 image frames per 1 second of video |
| `SNAPSHOTS_PER_BURST` | **5** | 5 CSI snapshots ↔ 1 image frame |
| `BURSTS_PER_WINDOW` | **5** | Model temporal window |
| `MAX_BURST_TO_FRAME_SYNC_ERROR_S` | **0.100** | Match tolerance at 5 FPS |

### D. Data currently on disk (3-day)

Approximate inventory:

| Item | Count |
|------|-------|
| CSI JSON files | 262 |
| Video files | 263 |
| Paired CSI+video sessions | 262 |
| Person sessions | 251 |
| Empty-room sessions | 11 |
| Extracted frame sessions | 263 |
| YOLO pose label JSONs | 252 (empty skipped) |
| Processed dataset | `data/processed/csi_pose_dataset.npz` |

### E. Train / val / test (current config)

**Person-ID split** (all three days of each ID stay together):

| Split | Person IDs | Role |
|-------|------------|------|
| **Train** | 001, 002, 003, 004, 005 | Learn pose + class (all days) |
| **Val** | 006 | Early stopping / LR schedule |
| **Test** | 007 | Final metrics + visualizations |

**Empty-room split** (by day folder, not by person):

| Day | Empty sessions go to |
|-----|----------------------|
| day1 | train |
| day2 | val |
| day3 | test |

Empty trains **classification only** (no keypoint/SAM pull to zeros).  
Live later: if `empty` → suppress skeleton.

Experiment tag: `EXPERIMENT_TAG = "day3_id007"`  
Model: `cnn_gru_multitask_sam`  
Classes: `sitting`, `standing`, `empty`  
`AUGMENT_CSI = True`  
`POSE_REPRESENTATION = "absolute"` (baseline for this run)

### F. Documentation completed

| File | Content |
|------|---------|
| `commands.txt` | Full runbook: prepare → split → train → eval → viz → live |
| `docs/CROSS_DAY_CSI_VARIATION.md` | Day-to-day CSI corr, why live fails, how to overcome |
| `docs/CHANGES_AND_RESULTS.md` | Older P0–P5 / LOSO / sitting failure analysis |
| `docs/PROJECT_STATUS.md` | **This file** — completed work + current run |
| `README.md` | Links to the docs above |

### G. Cross-day CSI finding (diagnostic, not a model metric)

Compared mean amplitude fingerprints (all usable 3×53 bins) with **Pearson correlation**:

| Class | Mean corr across days |
|-------|------------------------|
| Sitting | ~0.44 |
| Standing | ~0.45 |
| Empty | ~0.25 |

Meaning: same pose **label** ≠ same CSI **shape** across days.  
Within a day, sitting vs standing often look very similar in RF (corr often >0.9).  
Documented in `docs/CROSS_DAY_CSI_VARIATION.md`.

---

## 3. What is running right now

### Active process

```text
.venv/bin/python -m src.train
Experiment : day3_id007
Model      : cnn_gru_multitask_sam
```

Observed at last status check:

| Item | Value |
|------|-------|
| Training history epochs logged | ~27 |
| Best validation epoch so far | **12** (lowest val loss) |
| Checkpoints | `outputs/checkpoints/cnn_gru_multitask_sam_day3_id007_best.pt` |
| | `outputs/checkpoints/cnn_gru_multitask_sam_day3_id007_last.pt` |
| History file | `outputs/cnn_gru_multitask_sam_day3_id007_training_history.json` |

Training uses early stopping (`EARLY_STOPPING_PATIENCE = 20`) and may still run until that triggers or `NUM_EPOCHS = 120`.

### Not running yet (next after train finishes)

1. `python -m src.evaluate`  
2. `python -m src.evaluate_by_pose --split test`  
3. `python -m src.visualize ...` (skeleton overlays / prediction-only)  
4. `python -m src.live.infer --file data/raw/csi/day3/...` (including empty)

All of these are listed in order in `commands.txt`.

---

## 4. Working pipeline (what “running” means end-to-end)

```text
data/raw/videos/dayN/*.mp4
        → extract_frames @ 5 FPS
        → annotate_pose (YOLO, skip empty)
        → pose_labels

data/raw/csi/dayN/*.json
        → snapshots → bursts (5 CSI snaps)
        → sync to nearest frame
        → build_dataset NPZ
                ↓
        train (person split + empty-by-day)
                ↓
        evaluate / visualize skeletons
                ↓
        live infer: empty → no skeleton, else draw pose
```

---

## 5. How empty is trained and tested

| Stage | Behavior |
|-------|----------|
| Data | `000_empty_room*.json` (+ matching video); no YOLO GT |
| Train loss | **Classification only**; pose losses masked |
| Val | day2 empties (for early stopping with person val) |
| Test | day3 empties; report empty precision/recall |
| Live | If class=`empty` or `P(empty) > 0.5` → hide skeleton |

You **do** need empty **and** person data together for the 3-class head.

---

## 6. How to change train / test

Edit `src/config.py` only:

```python
FIXED_TEST_SUBJECTS = ("007",)  # who you evaluate / visualize
FIXED_VAL_SUBJECTS  = ("006",)  # who you validate on
EXPERIMENT_TAG = "day3_id007"   # change when split changes
```

Train = every remaining person ID across **all three days**.

Then re-run from split/train in `commands.txt` (rebuild dataset only if FPS/raw data changed).

---

## 7. Key paths

```text
Project root:
  ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline

Raw input:
  data/raw/csi/day{1,2,3}/
  data/raw/videos/day{1,2,3}/

Interim:
  data/interim/frames/
  data/interim/pose_labels/

Processed:
  data/processed/csi_pose_dataset.npz

Current run outputs:
  outputs/checkpoints/cnn_gru_multitask_sam_day3_id007_*.pt
  outputs/cnn_gru_multitask_sam_day3_id007_training_history.json

Archived old models:
  archive/previous_7subject_pipeline/checkpoints/
```

---

## 8. Commands reference

Full copy-paste sequence: **`commands.txt`**

Short path after train completes:

```bash
cd ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
source .venv/bin/activate
export PYTHONPATH=.

.venv/bin/python -m src.evaluate
.venv/bin/python -m src.evaluate_by_pose --split test
.venv/bin/python -m src.visualize --pose sitting --num-samples 20 --prediction-only
.venv/bin/python -m src.visualize --pose standing --num-samples 20 --prediction-only
.venv/bin/python -m src.live.infer --file data/raw/csi/day3/007_sitting_center_front.json
.venv/bin/python -m src.live.infer --file data/raw/csi/day3/000_empty_room.json
```

---

## 9. Bottom line

**Completed:** archive old models; 3-day data; 5 FPS / 5 CSI sync; train/eval for `day3_id007`.

**Results:** pose on test 007 is strong (PCK ~83%); **empty recall on day3 is 0%** (day1 empty memorized). Full write-up: `docs/RESULTS_DAY3_ID007.md`.

**Later success:** `day4_id005_emptyfix` — PCK ~88% on test 005; **empty 100%** on day1–4. See `docs/RESULTS_DAY4_ID005_EMPTYFIX.md`. Final package: `docs/FINAL_OUTPUT.md` + `models/cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt`.

**Next:** live inference with empty gating; optional TCN / sitting improvements.

## 10. Next day / next experiments

- Fix empty-room generalization; follow empty eval protocol in `docs/RESULTS_DAY3_ID007.md`.
- Try a new backbone (e.g. **TCN** instead of GRU): add `cnn_tcn_multitask_sam`, keep the same 3-class + SAM heads / day3 split, new `EXPERIMENT_TAG`, then retrain and compare to `day3_id007`.
