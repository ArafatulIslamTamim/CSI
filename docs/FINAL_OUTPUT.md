# Final Output — CSI Pose Model (day4_id005_emptyfix)

This document describes the **final delivered model** from the 4-day WiFi CSI human pose pipeline: where files live, how the model is configured, what metrics it achieved, and how to run evaluation / live inference.

Related docs:

- Full metrics write-up: [RESULTS_DAY4_ID005_EMPTYFIX.md](RESULTS_DAY4_ID005_EMPTYFIX.md)
- Earlier empty failure: [RESULTS_DAY3_ID007.md](RESULTS_DAY3_ID007.md)
- Cross-day CSI variation: [CROSS_DAY_CSI_VARIATION.md](CROSS_DAY_CSI_VARIATION.md)
- Project status: [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

## 1. What was delivered

A multitask CNN+GRU model that maps WiFi CSI bursts to:

1. **17 COCO keypoints** (skeleton), and  
2. **3-class labels**: `sitting` / `standing` / `empty`  

When the class is **empty**, live inference **suppresses** the skeleton.

| Item | Location |
|------|----------|
| **Saved good model (backup)** | `models/cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt` |
| Training checkpoint (same weights) | `outputs/checkpoints/cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt` |
| Last epoch checkpoint | `outputs/checkpoints/cnn_gru_multitask_sam_day4_id005_emptyfix_last.pt` |
| Test metrics | `outputs/evaluation/cnn_gru_multitask_sam_day4_id005_emptyfix_metrics.json` |
| Per-pose metrics | `outputs/evaluation/cnn_gru_multitask_sam_day4_id005_emptyfix_test_metrics_by_pose.json` |
| Test predictions | `outputs/evaluation/cnn_gru_multitask_sam_day4_id005_emptyfix_test_predictions.npz` |
| Training history | `outputs/cnn_gru_multitask_sam_day4_id005_emptyfix_training_history.json` |
| Dataset used | `data/processed/csi_pose_dataset.npz` |
| Experiment tag | `day4_id005_emptyfix` |

Checkpoint metadata: **epoch 4**, `model_name = cnn_gru_multitask_sam`, ~**514,565** trainable parameters.

---

## 2. Final metrics (test person 005)

| Metric | Value |
|--------|-------|
| Overall PCK@0.2 | **87.7%** |
| Sitting PCK@0.2 | **79.1%** |
| Standing PCK@0.2 | **96.5%** |
| MPJPE (norm / pixel) | 0.053 / ~41 px |
| Classification accuracy | **98.4%** |
| Empty recall | **100%** |
| Empty precision | **100%** |

Confusion (test): empty is clean (269/269 correct). Small sitting↔standing swaps only.

Empty domain check (same checkpoint): **100% empty accuracy on day1, day2, day3, and day4**.

This fixes the earlier `day3_id007` failure (0% empty on day2/day3).

---

## 3. Model configuration

### 3.1 Architecture

```text
CSI window [5 bursts × 5 snapshots × 3 Rx × 53 subcarriers]
        → CNN (32 → 64 → 128, dropout 0.3)
        → GRU (hidden 128, 1 layer)
        → SAM pose head (FC → 2×17×17 keypoints)
        → Class head (sitting / standing / empty)
```

| Setting | Value |
|---------|-------|
| `MODEL_NAME` | `cnn_gru_multitask_sam` |
| `POSE_REPRESENTATION` | `absolute` |
| `SAM_DECODER` | `fc` |
| `CNN_CHANNELS` | `(32, 64, 128)` |
| `CNN_FEATURE_DIM` | 128 |
| `CNN_DROPOUT` | 0.30 |
| `GRU_HIDDEN_SIZE` | 128 |
| `GRU_NUM_LAYERS` | 1 |
| `FC_HIDDEN` | 256 |
| `POSE_CLASSES` | `sitting`, `standing`, `empty` |
| Keypoints | 17 (COCO) |
| Bone / angle / GCN / adversarial | off |

### 3.2 CSI / video sync

| Setting | Value |
|---------|-------|
| Feature | amplitude |
| Antennas | 1 Tx × 3 Rx |
| Subcarriers | 53 usable (drop indices 27–37) |
| `TARGET_VIDEO_FPS` | **5** |
| `SNAPSHOTS_PER_BURST` | **5** (5 CSI snaps ↔ 1 image frame) |
| `BURSTS_PER_WINDOW` | **5** |
| Sync tolerance | 0.100 s |
| Normalization | `per_snapshot_subcarrier` + train mean/std |
| `AUGMENT_CSI` | True |

### 3.3 Data split

| Split | Content |
|-------|---------|
| **Train** | people **001, 002, 003, 004, 007** (days 1–4) |
| **Val** | person **006** |
| **Test** | person **005** |
| Empty train | 11 sessions across day1–4 |
| Empty val | `day2_000_empty_room_3` |
| Empty test | `day3_000_empty_room_3`, `day4_000_empty_room_3` |

Raw layout:

```text
data/raw/csi/day{1,2,3,4}/
data/raw/videos/day{1,2,3,4}/
```

### 3.4 Loss / optimizer

| Setting | Value |
|---------|-------|
| `LAMBDA_COORD` | 1.0 |
| `LAMBDA_RELATIVE` | 0.10 |
| `LAMBDA_CLASS` | **1.0** |
| `CLASS_LOSS_WEIGHTS` | `(1.0, 1.0, 20.0)` — empty ×20 |
| Batch size | 32 |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Max epochs | 120 |
| Early stopping patience | 20 |
| Best epoch selected | **4** (lowest val loss) |
| Device | CPU |

Empty samples: **classification only** (no keypoint/SAM pull to zeros).

---

## 4. Pipeline that produced this output

```text
Video (5 FPS frames) → YOLO-Pose GT (COCO-17)
CSI JSON → snapshots → bursts (5) → windows (5)
        → sync to frames → dataset NPZ
        → train CNN+GRU multitask
        → evaluate (PCK, class, empty)
        → live: empty → no skeleton; else draw pose
```

Teacher: Ultralytics YOLO-Pose (`yolo26n-pose.pt`).  
Pose labels: `data/interim/pose_labels/`  
YOLO viz: `data/interim/pose_visualizations/`

---

## 5. How to use the final model

### 5.1 Live inference (recommended)

With current `EXPERIMENT_TAG = "day4_id005_emptyfix"`, live loads  
`outputs/checkpoints/..._best.pt` automatically:

```bash
cd ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
source .venv/bin/activate
export PYTHONPATH=.

# Empty → skeleton suppressed
.venv/bin/python -m src.live.infer --file data/raw/csi/day4/000_empty_room.json

# Person → skeleton drawn
.venv/bin/python -m src.live.infer --file data/raw/csi/day4/005_sitting_center_front.json --show-video
.venv/bin/python -m src.live.infer --file data/raw/csi/day4/005_standing_center_front.json --show-video
```

Empty gate: class=`empty` or `P(empty) > 0.5` (`LIVE_EMPTY_PROB_THRESHOLD`).

### 5.2 Offline evaluate / visualize

```bash
.venv/bin/python -m src.evaluate
.venv/bin/python -m src.evaluate_by_pose --split test
.venv/bin/python -m src.visualize --pose sitting --num-samples 20 --prediction-only
.venv/bin/python -m src.visualize --pose standing --num-samples 20 --prediction-only
```

### 5.3 Load the backup in `models/` from code

```python
import torch
from src.models import build_model

ckpt = torch.load(
    "models/cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt",
    map_location="cpu",
    weights_only=False,
)
model = build_model()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
# ckpt["epoch"] == 4, ckpt["model_name"] == "cnn_gru_multitask_sam"
```

Keep `src/config.py` consistent with this experiment (`MODEL_NAME`, window sizes, 3 classes) when loading.

---

## 6. Known limits

- Sitting PCK (~79%) is lower than standing (~96%).  
- CSI cannot match camera-perfect 2D joints.  
- Results are for **held-out person 005**, not a full 7-fold LOSO table.  
- Antenna / room changes can still shift CSI; keep empty captures from multiple days if you retrain.

---

## 7. One-paragraph summary

> The final model `cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt` (epoch 4) was trained on 4 days of CSI/video with person-ID split (test **005**, val **006**), 5 FPS video, 5 CSI snapshots per frame, and multitask sitting/standing/empty heads with empty class weight ×20. On test person 005 it reaches **87.7% PCK@0.2** (sitting 79%, standing 97%) and **100% empty recall/precision**, with empty correct on all four days. Weights are saved under `models/` and `outputs/checkpoints/` for live inference with empty-room skeleton gating.
