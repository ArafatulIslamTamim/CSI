# Results: day3_id007 — Evaluation Report & Empty-Room Guidance

Experiment: **`day3_id007`**  
Model: `cnn_gru_multitask_sam`  
Checkpoint: epoch **12** (`outputs/checkpoints/cnn_gru_multitask_sam_day3_id007_best.pt`)  
Test person: **007** (all days) + day3 empty sessions  
Train people: 001–005 | Val person: 006  
Empty split used: day1→train, day2→val, day3→test  

Metric files:

- `outputs/evaluation/cnn_gru_multitask_sam_day3_id007_metrics.json`
- `outputs/evaluation/cnn_gru_multitask_sam_day3_id007_test_metrics_by_pose.json`
- `outputs/evaluation/cnn_gru_multitask_sam_day3_id007_test_predictions.npz`
- Training history: `outputs/cnn_gru_multitask_sam_day3_id007_training_history.json`

Related docs: `docs/CROSS_DAY_CSI_VARIATION.md`, `docs/PROJECT_STATUS.md`

---

## 1. Overall result (one glance)

| Area | Result | Verdict |
|------|--------|---------|
| Pose (PCK@0.2, non-empty) | **83.4%** | Good |
| Standing PCK | **94.1%** | Strong |
| Sitting PCK | **72.7%** | OK, weaker |
| Classification accuracy | **80.8%** | Misleading — driven by person classes |
| **Empty recall** | **0%** | **Failed** |
| Live empty gating | Not usable | Must fix before live “show nothing” |

**Bottom line:** CSI pose on held-out person 007 is usable for skeleton demos. Empty-room detection did **not** generalize beyond day1, so live empty gating will not work with this checkpoint.

---

## 2. Pose metrics (test, non-empty)

| Metric | Overall | Sitting | Standing |
|--------|---------|---------|----------|
| Samples | 4752 | 2374 | 2378 |
| PCK@0.2 (norm) | **0.834** | 0.727 | **0.941** |
| MPJPE (norm) | 0.059 | 0.063 | 0.055 |
| Pixel MPJPE | ~46 px | ~48 px | ~44 px |
| Class recall | — | 77.3% | 97.9% |

Sitting is the harder pose (more sitting→standing confusion: 539 of 2374). Standing is strong.

Training stopped at epoch 32 (early stopping); best val loss was epoch 12 — that checkpoint was evaluated.

---

## 3. Classification confusion (test)

Rows = true class, columns = predicted (`sitting`, `standing`, `empty`):

| True \ Pred | sitting | standing | empty |
|-------------|---------|----------|-------|
| sitting | 1835 | 539 | 0 |
| standing | 35 | 2327 | 16 |
| **empty** | **378** | **21** | **0** |

| Class | Support | Precision | Recall |
|-------|---------|-----------|--------|
| sitting | 2374 | 0.82 | 0.77 |
| standing | 2378 | 0.81 | 0.98 |
| **empty** | 399 | **0.00** | **0.00** |

Every empty test sample was called a person (94.7% sitting, 5.3% standing).

---

## 4. Empty-room evaluation (detailed)

### 4.1 Test empties (day3) — failed hard

| Item | Value |
|------|-------|
| Empty test samples | 399 (3 sessions) |
| Correct empty predictions | **0** |
| Mean P(sitting) | **0.925** |
| Mean P(standing) | 0.073 |
| Mean P(empty) | **0.002** |
| Max P(empty) | **0.010** |
| Count with P(empty) > 0.5 | **0** |
| Count with P(empty) > 0.1 | **0** |

Per session:

| Session | n | → sitting | → standing | → empty |
|---------|---|-----------|------------|---------|
| `day3_000_empty_room` | 133 | 119 | 14 | 0 |
| `day3_000_empty_room_2` | 132 | 131 | 1 | 0 |
| `day3_000_empty_room_3` | 134 | 128 | 6 | 0 |

Live rule `P(empty) > 0.5` would **never** suppress the skeleton on these day3 empties.

Pose PCK/MPJPE on empty are correctly `null` (no skeleton GT; empty uses classification only).

### 4.2 Domain check — why it failed

Same checkpoint scored on **all** empty days:

| Day | Split role | Empty accuracy | Mean P(empty) |
|-----|------------|----------------|---------------|
| **day1** | train | **100%** (675/675) | **0.999** |
| **day2** | val | **0%** | 0.005 |
| **day3** | test | **0%** | 0.002 |

**Diagnosis:** the model memorized **day1 empty CSI**. It did not learn a day-invariant “room empty” concept. Day2/day3 empty fingerprints differ (cross-day empty Pearson ~0.25; see `docs/CROSS_DAY_CSI_VARIATION.md`), so they look like sitting.

Contributing setup issues:

- Empty train only from day1 (**675** samples) vs ~**23.6k** person train samples  
- `LAMBDA_CLASS = 0.1` (weak vs pose loss)  
- Checkpoint chosen on total val loss, not empty recall  

---

## 5. Suggested empty-room evaluation protocol

Use this checklist for every future run (including after you fix training).

### 5.1 Metrics to always report

For the **empty** class on each of train / val / test (or per day):

1. **Recall** — of true empties, how many predicted empty? (main live metric)  
2. **Precision** — of predicted empties, how many truly empty?  
3. **Mean / median / max P(empty)** on true empty samples  
4. **Threshold sweep** — how many empties have P(empty) > {0.1, 0.3, 0.5, 0.7}  
5. **Confusion with sitting/standing** — where empties go when wrong  
6. **Per-session table** — each `000_empty_room*` file separately  

Do **not** use PCK/MPJPE for empty (no pose GT).

### 5.2 Required domain tests (not only “test split”)

Because empty RF shifts by day, always run:

| Test | Pass criteria (suggested) |
|------|---------------------------|
| Empty on **train days** | High recall (sanity; should not be 0) |
| Empty on **val days** | Recall ≫ 0; ideally ≥ 0.8 |
| Empty on **held-out day** | Recall ≥ 0.8 before trusting live |
| Person sitting/standing | Must not all become empty (check false empty rate) |

A model with day1 empty = 100% and day3 empty = 0% is a **fail** for live, even if overall accuracy looks fine.

### 5.3 Live-oriented evaluation

Before claiming live empty gating works:

```bash
# Replay empty from each day
.venv/bin/python -m src.live.infer --file data/raw/csi/day1/000_empty_room.json --speed 0 --no-display
.venv/bin/python -m src.live.infer --file data/raw/csi/day2/000_empty_room.json --speed 0 --no-display
.venv/bin/python -m src.live.infer --file data/raw/csi/day3/000_empty_room.json --speed 0 --no-display

# Replay a person capture (skeleton should appear)
.venv/bin/python -m src.live.infer --file data/raw/csi/day3/007_sitting_center_front.json --speed 0 --no-display
```

Pass if:

- Empty days: skeleton suppressed most of the time (`P(empty)` high)  
- Person: skeleton shown; not stuck on empty  

Also log class probabilities over time; do not rely only on the drawn skeleton.

### 5.4 Suggested acceptance bar (for your live goal)

Ship / use a checkpoint for live only if **all** hold:

| Gate | Suggested bar |
|------|----------------|
| Empty recall on held-out day | ≥ **0.80** |
| Mean P(empty) on held-out empty | ≥ **0.5** (or tune threshold after) |
| False empty rate on person test | ≤ **0.05** (few person frames called empty) |
| Sitting/standing PCK (optional) | Keep monitoring; pose can be secondary to gating |

Tune `LIVE_EMPTY_PROB_THRESHOLD` only **after** the model assigns real probability mass to empty on new days.

### 5.5 Quick commands after each retrain

```bash
cd ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
source .venv/bin/activate
export PYTHONPATH=.

.venv/bin/python -m src.evaluate
.venv/bin/python -m src.evaluate_by_pose --split test

# Inspect empty block in metrics
python - <<'PY'
import json
from src import config
m = json.load(open(config.METRICS_FILE))
print('empty', m['classification_report']['per_class']['empty'])
print('confusion', m['classification_report']['confusion_matrix'])
PY
```

Then re-run the **per-day empty domain check** (softmax on day1/day2/day3 empties), same as in §4.2.

---

## 6. How to fix empty (training), then re-evaluate

1. **Mix empty across days in train** (do not train empty on day1 only).  
2. **Up-weight empty** (`LAMBDA_CLASS` and/or class weights / oversampling).  
3. New `EXPERIMENT_TAG` (e.g. `day3_id007_emptyfix`).  
4. Retrain; select checkpoint using **val empty recall**, not only total val loss.  
5. Re-run §5 protocol; only then try live empty gating.

Data files themselves are fine (11 empty sessions, 1476 samples). The failure is **split + imbalance + day shift**, not missing empty captures.

---

## 7. Summary for notes / reports

> On `day3_id007`, overall PCK@0.2 was **83.4%** (standing **94%**, sitting **73%**) on person 007. Empty-room detection failed completely on day3 (**0% recall**); the model predicted sitting with mean probability **0.93**. The same checkpoint got **100%** empty accuracy on day1 train empties and **0%** on day2/day3 — evidence of memorizing day1 empty CSI. Before live inference, retrain with multi-day empty mixing and stronger empty classification, and evaluate empty recall **per day**, not only overall accuracy.
