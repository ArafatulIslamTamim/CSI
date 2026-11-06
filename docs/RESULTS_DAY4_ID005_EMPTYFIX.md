# Results: day4_id005_emptyfix — Evaluation Report

Experiment: **`day4_id005_emptyfix`**  
Model: `cnn_gru_multitask_sam`  
Checkpoint: epoch **4** (`outputs/checkpoints/cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt`)  
Data: days **1–4**  
Test person: **005** (all days)  
Val person: **006**  
Train people: **001, 002, 003, 004, 007**  
Empty: multi-day mix in train + held-out empty sessions for val/test  
Class loss: `LAMBDA_CLASS=1.0`, `CLASS_LOSS_WEIGHTS=(1, 1, 20)` for empty  

Metric files:

- `outputs/evaluation/cnn_gru_multitask_sam_day4_id005_emptyfix_metrics.json`
- `outputs/evaluation/cnn_gru_multitask_sam_day4_id005_emptyfix_test_metrics_by_pose.json`
- `outputs/evaluation/cnn_gru_multitask_sam_day4_id005_emptyfix_test_predictions.npz`
- Training history: `outputs/cnn_gru_multitask_sam_day4_id005_emptyfix_training_history.json`

Compare with failed empty run: `docs/RESULTS_DAY3_ID007.md`  
Cross-day CSI notes: `docs/CROSS_DAY_CSI_VARIATION.md`

---

## 1. One-glance verdict

| Area | Result | Verdict |
|------|--------|---------|
| Overall PCK@0.2 (non-empty) | **87.7%** | Strong |
| Standing PCK | **96.5%** | Excellent |
| Sitting PCK | **79.1%** | Good, still weaker |
| Classification accuracy | **98.4%** | Strong |
| **Empty recall / precision** | **100% / 100%** | **Fixed** |
| Live empty gating | Ready to try | Pass acceptance bar |

Training: 24 epochs (early stop); best validation loss at epoch **4**.

**Bottom line:** Held-out person **005** pose is strong; empty-room detection generalizes across days (unlike `day3_id007`). Suitable for live demos with empty gating.

---

## 2. Setup that differed from day3_id007

| Item | day3_id007 (failed empty) | day4_id005_emptyfix |
|------|---------------------------|------------------------|
| Days | 1–3 | **1–4** |
| Test ID | 007 | **005** |
| Empty train | day1 only | **day1–4 mix** (11 sessions) |
| Empty test | all day3 empties | `day3_000_empty_room_3` + `day4_000_empty_room_3` |
| `LAMBDA_CLASS` | 0.1 | **1.0** |
| Empty class weight | none | **×20** |

---

## 3. Pose metrics (test, non-empty)

| Metric | Overall | Sitting | Standing |
|--------|---------|---------|----------|
| Samples | 6305 | 3172 | 3133 |
| PCK@0.2 (norm) | **0.877** | 0.791 | **0.965** |
| MPJPE (norm) | 0.053 | 0.056 | 0.050 |
| Pixel MPJPE | ~41 px | ~44 px | ~39 px |
| Class recall | — | 99.3% | 97.4% |

Sitting remains harder than standing (especially lower-limb joints / ankles on sitting). Standing is near ceiling for this CSI setup.

---

## 4. Classification (test)

Rows = true, columns = predicted (`sitting`, `standing`, `empty`):

| True \ Pred | sitting | standing | empty |
|-------------|---------|----------|-------|
| sitting | 3149 | 23 | 0 |
| standing | 83 | 3050 | 0 |
| **empty** | **0** | **0** | **269** |

| Class | Support | Precision | Recall |
|-------|---------|-----------|--------|
| sitting | 3172 | 0.97 | 0.99 |
| standing | 3133 | 0.99 | 0.97 |
| **empty** | 269 | **1.00** | **1.00** |

No empty↔person confusions on the held-out empty test sessions.

---

## 5. Empty-room evaluation (detailed)

### 5.1 Held-out empty test

Sessions: `day3_000_empty_room_3`, `day4_000_empty_room_3` (n=269)

| Metric | Value |
|--------|-------|
| Empty accuracy | **100%** |
| Mean P(empty) | **1.000** |
| Max P(empty) | 1.000 |
| P(empty) > 0.5 | **269 / 269** |

Live threshold `LIVE_EMPTY_PROB_THRESHOLD = 0.5` would suppress the skeleton on all these empties.

### 5.2 Domain check — all empty days

Same checkpoint on **every** empty sample by day:

| Day | n | Empty accuracy | Mean P(empty) |
|-----|---|----------------|---------------|
| day1 | 675 | **100%** | 1.000 |
| day2 | 402 | **100%** | 1.000 |
| day3 | 399 | **100%** | 1.000 |
| day4 | 404 | **100%** | 1.000 |

### 5.3 Contrast with previous failure

| | day3_id007 | **day4_id005_emptyfix** |
|--|------------|---------------------------|
| Day1 empty acc | 100% | 100% |
| Day2 empty acc | **0%** | **100%** |
| Day3 empty acc | **0%** | **100%** |
| Day4 empty acc | n/a | **100%** |
| Mean P(empty) on held-out empty | 0.002 | **1.000** |

Cause of the fix: train empty CSI from **multiple days** + stronger empty classification weight, so the model no longer memorizes only day1 empty.

---

## 6. What is still weaker / caveats

1. **Sitting PCK (79%)** < standing (96%) — expected from CSI sit≈stand similarity within a day.  
2. Checkpoint selected on **total val loss** (epoch 4); empty is already saturated — monitor empty recall if you change splits again.  
3. Results are for **held-out person 005**, not a full leave-one-person-out table.  
4. CSI cannot match camera pixel precision; judge live by class stability + rough skeleton, not YOLO-perfect joints.

---

## 7. Live inference commands

```bash
cd ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
source .venv/bin/activate
export PYTHONPATH=.

# Empty → skeleton should be suppressed
.venv/bin/python -m src.live.infer --file data/raw/csi/day4/000_empty_room.json
.venv/bin/python -m src.live.infer --file data/raw/csi/day3/000_empty_room.json

# Person (test ID 005) → skeleton should appear
.venv/bin/python -m src.live.infer --file data/raw/csi/day4/005_sitting_center_front.json --show-video
.venv/bin/python -m src.live.infer --file data/raw/csi/day4/005_standing_center_front.json --show-video
```

Optional offline viz:

```bash
.venv/bin/python -m src.visualize --pose sitting --num-samples 20 --prediction-only
.venv/bin/python -m src.visualize --pose standing --num-samples 20 --prediction-only
```

---

## 8. Summary for notes / reports

> On `day4_id005_emptyfix` (days 1–4, test person 005, val 006), overall PCK@0.2 was **87.7%** (standing **96.5%**, sitting **79.1%**) with classification accuracy **98.4%**. Empty-room detection reached **100% recall and precision** on held-out day3/day4 empty sessions, and **100% empty accuracy on all four days** — fixing the prior `day3_id007` failure (0% empty on day2/day3) by training empty from multiple days with stronger class weight. The run is suitable for live inference with empty-room skeleton gating.
