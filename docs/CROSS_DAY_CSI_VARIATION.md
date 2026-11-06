# Cross-Day CSI Variation — What It Means and How to Overcome It

This note documents measured differences between **day1 / day2 / day3**
CSI captures in this project, why that hurts pose / live inference, and
what to do about it.

Related layout:

- Raw data: `data/raw/csi/day{1,2,3}/`, `data/raw/videos/day{1,2,3}/`
- Split / training commands: `commands.txt`
- Experiment narrative: `docs/CHANGES_AND_RESULTS.md`

---

## 1. What we compared

For each capture we built a **CSI amplitude fingerprint**:

1. Parse complete CSI snapshots (3 Rx × 53 usable subcarriers).
2. Average amplitudes over many snapshots in that file.
3. Flatten to one vector of length \(3 \times 53 = 159\).

Then we compared the **same filename** across days, e.g.:

- `day1/001_sitting_center_front.json`
- `day2/001_sitting_center_front.json`
- `day3/001_sitting_center_front.json`

### Metric: `corr` (Pearson correlation)

| Corr | Meaning |
|------|---------|
| **+1.0** | Same shape: days match closely |
| **~0.8–0.9** | Quite similar; mild day drift |
| **~0.4–0.5** | Only partly related; large RF change |
| **~0** | Almost unrelated |
| **Negative** | Often opposite pattern; days look very different |

`corr` is **not** accuracy. It only says whether two CSI profiles have a
similar shape (mean-centered).

### Metric: cosine similarity

\[
\text{cosine}(x,y) = \frac{x\cdot y}{\|x\|\,\|y\|}
\]

Unlike Pearson, cosine does **not** subtract the mean. Amplitudes are
always ≥ 0 and share a large common level, so **cosine is often very high
(~0.95+) even when Pearson is low or negative**. Use Pearson for “did the
subcarrier *shape* change?”; use cosine as a secondary check of overall
direction in amplitude space.

We also used **relL2%**: relative size of the difference between the two
fingerprints (higher = bigger change).

### Per-Rx analysis

Hardware uses **3 receivers** (code indices 0,1,2). In tables below they
are labeled **Rx1 / Rx2 / Rx3**. For each Rx we compare only that Rx’s
53-bin mean amplitude vector across days. `all` = flatten all 3×53 bins.

Raw numbers: `outputs/qa/cross_day_rx_similarity.json`

---

## 2. Measured results (this dataset)

### 2.1 Same pose, different days (all Rx together)

Across all paired sessions present on at least two days:

| Class | Mean Pearson | Median Pearson | Worst Pearson | Mean cosine |
|-------|--------------|----------------|---------------|-------------|
| Sitting | **0.44** | 0.66 | **-0.86** | **0.97** |
| Standing | **0.45** | 0.63 | **-0.65** | **0.97** |
| Empty room | **0.25** | 0.15 | **-0.22** | **0.90** |

**Interpretation:**

- **Pearson** says the fingerprint *shape* often changes a lot across days.
- **Cosine** stays high because amplitudes are positive and share a common
  scale — days are not “random noise,” but the **relative pattern across
  subcarriers** (what Pearson sees) still drifts. That drift is what hurts
  the pose/class model.

### 2.1b Per receiver (Rx1 / Rx2 / Rx3)

Mean similarity across the same day-pairs, **per Rx** (53 bins each):

**Sitting** (n=126 pairs)

| Scope | Mean Pearson | Mean cosine |
|-------|--------------|-------------|
| all (3×53) | 0.44 | 0.97 |
| **Rx1** | **-0.07** | 0.99 |
| **Rx2** | 0.48 | 0.99 |
| **Rx3** | 0.38 | 0.98 |

**Standing** (n=124 pairs)

| Scope | Mean Pearson | Mean cosine |
|-------|--------------|-------------|
| all (3×53) | 0.45 | 0.97 |
| **Rx1** | 0.28 | 1.00 |
| **Rx2** | 0.39 | 0.99 |
| **Rx3** | 0.45 | 0.99 |

**Empty** (n=9 pairs)

| Scope | Mean Pearson | Mean cosine |
|-------|--------------|-------------|
| all (3×53) | 0.25 | 0.90 |
| **Rx1** | **-0.43** | 0.90 |
| **Rx2** | **-0.29** | 0.97 |
| **Rx3** | 0.19 | 1.00 |

**Takeaways:**

1. Day shift is **not equal on every Rx** — e.g. sitting **Rx1** mean Pearson
   is near **0 / negative**, while Rx2 is more stable (~0.48).
2. Empty-room change is harsh on **Rx1 and Rx2** (negative mean Pearson).
3. High cosine on every Rx does **not** mean “no problem for training”;
   trust **Pearson (shape)** for domain-shift diagnosis.
4. The model still uses **all 3 Rx × 53** together; per-Rx tables are for
   understanding which antenna is least stable across days.

### 2.1c Example rows (Pearson vs cosine, per Rx)

`001_sitting_center_front` (hard):

| Day pair | P_all | C_all | P_Rx1 | C_Rx1 | P_Rx2 | C_Rx2 | P_Rx3 | C_Rx3 |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|
| day1–day2 | **-0.70** | 0.82 | +0.99 | 1.00 | +0.54 | 1.00 | **-0.98** | 0.81 |
| day1–day3 | **-0.60** | 0.84 | **-0.84** | 0.97 | +0.99 | 1.00 | +0.91 | 0.90 |
| day2–day3 | +0.18 | 0.97 | **-0.82** | 0.98 | +0.62 | 1.00 | **-0.82** | 0.99 |

`000_empty_room`:

| Day pair | P_all | C_all | P_Rx1 | P_Rx2 | P_Rx3 |
|----------|-------|-------|-------|-------|-------|
| day1–day2 | +0.69 | 0.97 | +0.15 | **-0.75** | +0.49 |
| day1–day3 | **-0.22** | 0.86 | **-0.69** | **-0.98** | +0.47 |
| day2–day3 | +0.12 | 0.85 | **-0.76** | +0.87 | **-0.50** |

Same file name, different days: some Rx stay aligned, others flip — that
is why live empty/pose gating can break on a new day.

### 2.2 Example: subject 001 (hard)

| Session | Day pair | Corr | relL2% |
|---------|----------|------|--------|
| Sitting center | day1–day2 | **-0.69** | 67% |
| Sitting center | day1–day3 | **-0.59** | 62% |
| Sitting center | day2–day3 | 0.15 | 26% |
| Standing center | day1–day2 | **-0.61** | 67% |
| Standing center | day1–day3 | 0.95 | 14% |

Some day pairs are close; others are almost inverted. Variation is
**person- and day-dependent**, not uniform.

### 2.3 Example: subject 007 (easier)

Sitting day-pairs often stay around corr **0.80–0.88**. Standing is more
mixed (~0.45–0.82). Easier people/days exist — do not judge the system
only on the worst ID.

### 2.4 Same day: sitting vs standing (pose separability)

Same person, same day, sitting vs standing center:

| ID | Day | Sit–stand corr |
|----|-----|----------------|
| 001 | day1 | **0.96** |
| 001 | day2 | **0.98** |
| 003 | day1 | **0.99** |
| 007 | day1 | **0.94** |

**Interpretation:** within one day, sitting and standing CSI often look
almost the same. **Day shift can be larger than pose shift.** That makes
classification and pose regression hard even before live inference.

### 2.5 Empty vs person

Empty vs a standing person on the same day is sometimes well separated,
sometimes not. Example: on **day2**, empty vs `001` standing was corr
**~0.97** for the mean fingerprint — nearly indistinguishable. That
breaks “show nothing when empty” if the model overfits one day’s empty.

RSSI also drifts (example empty means roughly **-49 → -54 → -59 dB**
across day1→day2→day3), which is environment / hardware / placement
drift, not a pose-label change.

---

## 3. What this means for training and live inference

### You are not collecting “duplicate” days

Three days are useful because CSI **really changes**. That variation is
the point — but only if training **sees** it.

### Why live inference fails under this variation

1. **Domain shift** — live CSI is “today’s channel.” If it looks like an
   unseen day (negative corr vs train days), class and joints go wrong.
2. **Pose confusion** — sit ≈ stand in RF → wrong class + wrong skeleton.
3. **Empty gating failure** — empty fingerprint changes across days →
   skeleton shown when empty, or hidden when occupied.
4. **Normalization mismatch** — live uses training CSI mean/std; a new
   day’s amplitude scale can sit out of distribution.
5. **Hard IDs** — people like 001 show the worst cross-day mismatch;
   average metrics hide this.

CSI cannot encode precise 2D image coordinates the way a camera can.
Even a good model will be less stable than YOLO-Pose video GT.

---

## 4. How to overcome it

### 4.1 Data / split strategy (do this first)

| Do | Don’t |
|----|--------|
| Train on **all days** of each training person | Train only day1, expect day3/live to work |
| Split by **person ID** (val/test people), keep all their days together | Split by day as the main live plan |
| Keep **empty captures on every day** in the dataset | Train empty from only one day |
| Evaluate per person **and** check hard days | Only report mean PCK |

**Recommended default in this repo** (`src/config.py`):

```text
Train = IDs 001..005  (day1 + day2 + day3 each)
Val   = ID 006
Test  = ID 007
Empty = day1 → train, day2 → val, day3 → test   (EMPTY_DAY_SPLIT)
```

Days are **natural augmentation**. Person IDs are the generalization
target (and a proxy for “new body / placement” at live time).

Change held-out people in `src/config.py`:

```python
FIXED_TEST_SUBJECTS = ("007",)
FIXED_VAL_SUBJECTS  = ("006",)
EXPERIMENT_TAG = "day3_id007"   # change tag when you change split
```

### 4.2 Training recipe for high variation

1. **Mix days in train** (already true with person-ID split).
2. **Enable CSI augmentation** for robustness runs:

   ```python
   AUGMENT_CSI = True
   ```

   (noise, Rx gain jitter, subcarrier dropout — see `src/dataset.py`.)

3. **Keep the 3-class head** (`sitting`, `standing`, `empty`) so live can
   suppress the skeleton when empty.
4. Start from a **simple strong baseline** (`cnn_gru_multitask_sam`) before
   stacking many geometry / adversarial extras.
5. Select / report models using **pose metrics and empty metrics**, not
   only classification-dominated validation loss.
6. After training, replay live on **all three days** of the test ID and on
   empty — not only one easy file.

Commands: see `commands.txt`.

### 4.3 Sync settings that match this collection

Configured for this 3-day pipeline:

| Setting | Value | Role |
|---------|-------|------|
| `TARGET_VIDEO_FPS` | 5 | 5 image frames per second |
| `SNAPSHOTS_PER_BURST` | 5 | 5 CSI snapshots ↔ 1 image frame |
| `BURSTS_PER_WINDOW` | 5 | Temporal context for the model |

Cross-day RF shift is separate from FPS; fixing FPS does not remove day
domain shift, but it keeps video↔CSI pairing consistent.

### 4.4 Live inference mitigations

1. Train with empty from multiple days (required).
2. Use the checkpoint whose `EXPERIMENT_TAG` matches the run you trust.
3. Tune `LIVE_EMPTY_PROB_THRESHOLD` if empty flickers
   (`src/config.py`).
4. Optional: adaptive / test-time CSI normalization (`evaluate --adapt-norm`
   style) so today’s scale matches better — useful when live RSSI/amplitude
   drifts.
5. Log class probabilities over time (`P(sitting)`, `P(standing)`,
   `P(empty)`) to see whether failure is class confusion vs joint error.
6. Keep antennas, AP, and room layout as stable as possible between
   train days and live demos; small physical changes dominate CSI.

### 4.5 What will not fix day variation alone

- Bone / angle losses (geometry) — make limbs more plausible, do not fix
  a shifted RF domain.
- More YOLO confidence — GT can be fine while CSI still drifts.
- Expecting all “sitting” files to share one CSI barcode — measurements
  show they do not.

Geometry and root-relative heads can still help **pose quality** once the
features are in-domain; they are not a substitute for multi-day training
and empty coverage.

### 4.6 If you can collect more data

Prioritize:

1. More **empty** on different days / times of day.
2. More **sitting** diversity (chairs, distance) on hard people.
3. Occasional **same script** repeated after moving furniture slightly —
   explicit domain coverage.
4. A short **calibration empty + standing** clip on the live day if you
   add adaptation later.

---

## 5. Practical checklist before blaming the model

- [ ] Train set includes day1+day2+day3 for training IDs  
- [ ] Empty exists on all days used in train/val/test  
- [ ] `AUGMENT_CSI = True` for the robustness experiment  
- [ ] Test metrics broken down by sitting / standing / empty  
- [ ] Live replay tried on day1, day2, and day3 files for the test ID  
- [ ] Live empty replay shows no skeleton  
- [ ] Checkpoint path matches current `EXPERIMENT_TAG`  
- [ ] Antennas / room not heavily changed since training captures  

---

## 6. Bottom line

- **Corr ~0.44–0.45** across days for sitting/standing means the same pose
  label does **not** imply the same CSI pattern.
- **Empty corr ~0.25** means the “no person” class also moves day to day.
- **Sit≈stand within a day** means pose separation in RF is weak.
- **Overcome it** by training on all days, splitting by person ID, keeping
  multi-day empty, using CSI augmentation, evaluating hard people/days,
  and treating live as a domain-shift problem — not a missing drawing bug.

For run commands and split editing, use `commands.txt`.
For older LOSO / P1–P5 experiment history, see `docs/CHANGES_AND_RESULTS.md`.
