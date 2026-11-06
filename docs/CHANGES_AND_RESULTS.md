# Changes, Results, and Why Sitting Fails

This document records **what you implemented** (experiment plan todos),
**what the measured results show so far**, and **why sitting predictions
are still wrong** for hard subjects (especially subject `001`).

---

## 0. What you did (experiment plan — completed code)

These are the six completed todos from your CSI pose experiment plan.
Full 7-fold LOSO after every stage (the 7th todo) was **started** but
**stopped mid-run** (see §2).

| # | Todo | Status | What you added |
|---|------|--------|----------------|
| 1 | Quantify YOLO GT quality on sitting; add MPJPE + per-pose per-fold eval | Done | `scripts/analyze_gt_quality.py` → `outputs/qa/gt_quality_report.json` + low-conf sitting spot-checks in `outputs/qa/gt_lowconf_sitting/`. Evaluation now reports **normalized + pixel MPJPE** alongside PCK (`src/metrics.py`, `src/evaluate.py`, `src/evaluate_by_pose.py`). |
| 2 | Root-relative + scale-decomposed targets + separate root head (P1) | Done | `POSE_REPRESENTATION = "root_relative"`: SAM diagonal stores `(xy - root) / scale`; model predicts a **root/scale head** separately from pose (`src/models.py`, `src/losses.py`, `src/dataset.py` / build path). Experiment: `rootrel`. |
| 3 | Bone-length + joint-angle losses (P2) | Done | `src/geometry.py` (`BONES`, `ANGLE_TRIPLETS`, `bone_lengths`, `joint_cosines`) + `LAMBDA_BONE` / `LAMBDA_ANGLE` in `src/losses.py`. Experiments: `rootrel_geo`, `geo_only`. |
| 4 | CSI augmentation + AdaBN-style test-time norm (P3) | Done | Training CSI noise/scale/Rx dropout via `AUGMENT_CSI` in `src/dataset.py`. Eval flag `--adapt-norm` recomputes global CSI mean/std on the test split (`compute_adaptive_stats`). Experiment: `rootrel_geo_aug` (+ `--adapt-norm-eval` in LOSO runner). |
| 5 | Subject-adversarial / gradient reversal (P4) | Done | `GradientReversal` + subject classifier on shared features (`src/models.py`); `ADVERSARIAL_SUBJECT` + adversarial loss weight in training. Experiment: `rootrel_geo_aug_adv`. |
| 6 | GCN decoder ablation vs loss-based constraints (P5) | Done | `GCNDecoder` over COCO skeleton adjacency (`src/models.py`, `SAM_DECODER = "gcn"`). Experiment: `rootrel_gcn_aug` (architecture prior) vs P2 (loss prior). |
| 7 | Re-run full 7-fold LOSO after each change; track worst-subject metrics | **Partial** | `scripts/run_loso_experiment.py` + `scripts/aggregate_loso.py`. Completed: all folds for `baseline3c` and `rootrel`; `rootrel_geo` only through subject **001** (then stopped). Later stages not fully LOSO’d yet. |

**How you run stages (does not rewrite `config.py`):**

```bash
python scripts/run_loso_experiment.py --experiment rootrel_geo --subjects 001
# names: baseline3c | rootrel | rootrel_geo | geo_only |
#        rootrel_geo_aug | rootrel_geo_aug_adv | rootrel_gcn_aug
```

Also in the same project (related earlier work): **empty-room as a 3rd class**
and **live replay inference** (`src/live/`). Those are separate from the P1–P5
todos but are part of the current pipeline.

---

## 1. Pipeline pieces (where the code lives)

### A. Empty-room class + live inference

| Area | Files | What |
|------|-------|------|
| 3-class labels | `src/config.py`, `src/build_dataset.py`, `src/dataset.py` | Classes `sitting`, `standing`, `empty`. Empty from `000_empty_room*.json`. |
| Loss masking | `src/losses.py` | Empty → classification only (no coordinate/SAM pull to zeros). |
| Live path | `src/live/` | Replay JSON → streaming bursts → model → empty gate → OpenCV. |

```text
CSI records → snapshots → bursts → 5-burst window
         → norms → CNN+GRU → class + keypoints
         → empty? suppress skeleton : show skeleton
```

### B. Your P0–P5 code map

| Stage | Key files / flags |
|-------|-------------------|
| P0 GT QA | `scripts/analyze_gt_quality.py`, `outputs/qa/` |
| P1 root-relative | `POSE_REPRESENTATION`, root head in `models.py` |
| P2 geometry loss | `geometry.py`, `LAMBDA_BONE`, `LAMBDA_ANGLE` |
| P3 aug + AdaBN | `AUGMENT_CSI`, `evaluate --adapt-norm` |
| P4 adversarial | `ADVERSARIAL_SUBJECT`, GRL in `models.py` |
| P5 GCN | `SAM_DECODER="gcn"`, `GCNDecoder` |
| LOSO runner | `scripts/run_loso_experiment.py` |

---

## 2. Results so far

### 2.1 YOLO GT quality (P0)

From `outputs/qa/gt_quality_report.json`:

- Sitting and standing both have **high** mean keypoint confidence (~0.98).
- So sitting failure is **not** mainly “YOLO totally broken,” though low-conf
  spot-check images were saved for visual review under
  `outputs/qa/gt_lowconf_sitting/`.
- Bone-length CV within subject+pose is still worth watching as a noise flag
  (see report per subject).

### 2.2 Subject `001` (hardest fold) — experiments that finished

| Experiment | Overall PCK@0.2 | MPJPE (norm) | Class acc | Sitting PCK | Standing PCK | Empty |
|------------|-----------------|--------------|-----------|-------------|--------------|-------|
| baseline3c | 49.0% | 0.130 | 64.7% | 43.2% | 54.8% | 100% |
| rootrel (P1) | 55.3% | 0.119 | 65.8% | 45.6% | 65.1% | 100% |
| rootrel_geo (P1+P2) | **56.8%** | **0.116** | **67.4%** | 44.0% | **69.7%** | **100%** |

Files:

- `outputs/evaluation/cnn_gru_multitask_sam_*_test001_metrics.json`
- Visualizations: `outputs/visualizations/rootrel_geo_test001/`
  (green = YOLO GT, red = CSI prediction)

**Takeaways for 001:**

- Empty detection is solved (~100%).
- P1 and P2 help **overall** and **standing**.
- **Sitting barely moves**; sitting class recall stays ~37–43%.
- Typical sitting viz: red skeleton shifted **up** (root/position error).

### 2.3 LOSO status

| Experiment | Folds finished |
|------------|----------------|
| baseline3c (+ AdaBN eval) | 001–007 |
| rootrel (P1) | 001–007 |
| rootrel_geo (P1+P2) | **001 only** (stopped during 002) |
| rootrel_geo_aug / adv / geo_only / gcn | Not finished |

On easier folds, baseline/rootrel often land ~80–95% PCK; **001/002** remain
the research bottleneck. Always report **worst-subject**, not only the mean.

---

## 3. Why sitting output is wrong

Your P1–P2 ideas were reasonable and partially help, but sitting on 001 stays
wrong for reasons that go beyond missing bone/angle terms.

### 3.1 Cross-subject CSI domain shift (main)

When the **classifier** fails on unseen sitting (~37% recall), features do not
transfer. Geometry loss cannot fix a wrong latent.

Sitting CSI is brittle: lower RF cross-section, little lower-body motion,
more posture/chair variability than standing.

### 3.2 Root / vertical bias

Overlays show the predicted skeleton floating above the person. That is a
**global position** error; every joint’s PCK collapses even if limb angles
look plausible. P1 was meant to separate root from shape — it helped standing
more than sitting on 001.

### 3.3 Geometry losses = plausibility, not accuracy

Bone/angle terms stop impossible limbs. They do **not** pin the skeleton to
the correct pixels. A “nice” sitting pose drawn too high still looks wrong.

### 3.4 Data / labels / size

- High YOLO confidence overall, but sitting can still be awkward (occlusion).
- Only ~6 train subjects under LOSO; sitting diversity is small.
- CSI does not encode precise 2D image coordinates.

### 3.5 What “wrong sitting” looks like

| Symptom | Meaning |
|---------|---------|
| Red skeleton above the body | Root / vertical localization error |
| Wrong scale / straight knees | Shape under-constrained; P2 only partially helps |
| Predicted standing while sitting | Feature-level confusion |
| Empty correct | Empty CSI is easy; person pose is hard |

---

## 4. What to do next

1. Finish LOSO for `rootrel_geo`, then `rootrel_geo_aug`, `rootrel_geo_aug_adv`,
   `geo_only`, `rootrel_gcn_aug`; aggregate with `scripts/aggregate_loso.py`
   and highlight **worst subject**.
2. Select checkpoints on coordinate/PCK, not classification-dominated val loss.
3. More sitting subjects / chairs / distances if you can collect data.
4. Keep using GT QA spot-checks; down-weight bad frames if needed.
5. Paper story: empty gating + ablations for **cross-subject sitting** failure.

---

## 5. Commands

```bash
cd ~/Downloads/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
source .venv/bin/activate
export PYTHONPATH=.
```

**GT quality (P0):**

```bash
python scripts/analyze_gt_quality.py
```

**Visualize subject 001 (P1+P2):**

```bash
cat > /tmp/viz_rootrel_geo_test001.json <<'EOF'
{"EXPERIMENT_TAG":"rootrel_geo_test001","POSE_REPRESENTATION":"root_relative","LAMBDA_BONE":0.1,"LAMBDA_ANGLE":0.1,"FIXED_TEST_SUBJECTS":["001"],"FIXED_VAL_SUBJECTS":["007"]}
EOF
export CSI_CONFIG_OVERRIDES=/tmp/viz_rootrel_geo_test001.json
python -m src.visualize --pose sitting --num-samples 20 --background original
python -m src.visualize --pose standing --num-samples 20 --background original
```

**Resume LOSO:**

```bash
python scripts/run_loso_experiment.py --experiment rootrel_geo --subjects 002 003 004 005 006 007
```

**Live replay:**

```bash
python -m src.live.infer --file data/raw/csi/001_sitting_center_front.json
python -m src.live.infer --file data/raw/csi/000_empty_room_5.json
```

---

## 6. Bottom line

- **You implemented:** P0 GT QA + MPJPE; P1 root-relative representation;
  P2 bone/angle loss; P3 aug/AdaBN; P4 subject-adversarial; P5 GCN decoder;
  LOSO runner. Empty class + live inference are also in the repo.
- **Works well:** empty detection; standing on easier subjects; P1/P2 help 001
  standing / overall a bit.
- **Still wrong:** sitting on hard subjects (esp. 001) — mainly **cross-subject
  CSI shift + root/position error**, not “we forgot anatomical constraints.”
- **Open:** finish full LOSO for P2+ later stages and report worst-subject tables.
