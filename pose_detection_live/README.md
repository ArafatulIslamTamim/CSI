# CSI Live Pose Detection (cnn_gru_multitask_sam)

Real-time WiFi CSI → pose using your trained multitask SAM checkpoint
(`sitting` / `standing` / `empty` + 17 keypoints).

Default weights in this folder:

```text
pose_detection_live/best_model.pt
```

(= `models/cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt`)

Also needs train-set CSI stats from the project root:

```text
outputs/csi_normalization_stats_day4_id005_emptyfix.npz
```

---

## Quick start (GUI)

```bash
cd ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
source .venv/bin/activate
cd pose_detection_live
python3 live_gui.py
```

Or with an explicit checkpoint:

```bash
python3 live_gui.py --checkpoint best_model.pt
```

---

## CLI replay (no BPI)

```bash
cd ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
source .venv/bin/activate
cd pose_detection_live

python3 run_live.py --replay ../data/raw/csi/day4/000_empty_room.json
python3 run_live.py --replay ../data/raw/csi/day4/005_sitting_center_front.json
python3 run_live.py --replay ../data/raw/csi/day4/005_standing_center_front.json
```

---

## CLI live (Banana Pi SSH)

```bash
python3 run_live.py \
  --host 192.168.1.219 \
  --user root \
  --password YOUR_PASS \
  --wifi-if wifi0 \
  --duration 30
```

---

## How it works

| Step | Module | Description |
|------|--------|-------------|
| CSI capture | `bpi_ssh_worker.py` | SSH poll `mt76-vendor dump csi` |
| Preprocessing | `csi_preprocessor.py` | Same as training: bursts → `[5,5,3,53]` + local/z-score norm |
| Inference | `pose_engine.py` | Loads `cnn_gru_multitask_sam` + SAM → keypoints |
| Visualization | `skeleton_viz.py` | Draw 17 joints (blank for `empty`) |

Input must match training: **5 bursts × 5 snapshots × 3 RX × 53 usable subcarriers**.

---

## Refresh the checkpoint

```bash
cd ~/Downloads/CSI/csi_pose_yolopose_full_pipeline/csi_pose_yolopose_pipeline
cp models/cnn_gru_multitask_sam_day4_id005_emptyfix_best.pt \
   pose_detection_live/best_model.pt
```
