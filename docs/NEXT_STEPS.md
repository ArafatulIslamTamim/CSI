# Next steps after pipeline upgrade

## What was implemented

1. **Backup restore** — `bash scripts/restore_backup.sh` → `day5_id005_l1l2` in `pose_detection_live/best_model.pt`
2. **Day split (sit/stand only, no empty)** — train day1+2+5, val day3, test day4
3. **Checkpoint selection** — `CHECKPOINT_SELECTION=val_classification_accuracy` (not total loss)
4. **500 ms causal windows** — `WINDOW_MODE=time_causal`, shape `[32,3,53]`
5. **TCN model** — `tcn_multitask_sam`, experiment `day125_val3_test4_tcn_v1`
6. **Parity script** — `scripts/compare_preprocess_parity.py`

## Commands

- TCN train/eval: see `commands_tcn_daysplit.txt`
- Restore 3-class model: `bash scripts/restore_backup.sh`
- Parity:  
  `PYTHONPATH=. python scripts/compare_preprocess_parity.py --csi data/raw/csi/day5/005_standing_center_front.json --session day5_005_standing_center_front`

## Parity note

Training windows end at **video label times**; live emits a window on **every CSI snapshot** (~10 ms).  
Windows matched by nearest timestamp can still differ slightly.  
For strict parity, compare offline rebuild at label `end_ts` vs live window at the **same** `end_ts` (future improvement).

## After training finishes

```bash
TAG=day125_val3_test4_tcn_v1
cp outputs/checkpoints/tcn_multitask_sam_${TAG}_best.pt pose_detection_live/best_model.pt
cd pose_detection_live
../.venv/bin/python3 run_live.py --replay ../data/raw/csi/day4/005_sitting_center_front.json
```

Monitor training: `tail -f outputs/train_day125_tcn_v1.log`
