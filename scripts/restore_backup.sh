#!/usr/bin/env bash
# Restore the backed-up day5_id005_l1l2 3-class model for in-domain replay/GUI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../" && pwd)"
cd "$ROOT"

TAG_OLD=day5_id005_l1l2
BACKUP="${BACKUP:-backups/${TAG_OLD}_20250828}"

if [[ ! -d "$BACKUP" ]]; then
  # Try latest backup folder for this tag.
  BACKUP="$(ls -dt backups/${TAG_OLD}_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "$BACKUP" || ! -d "$BACKUP" ]]; then
  echo "No backup found. Expected backups/${TAG_OLD}_YYYYMMDD/"
  exit 1
fi

echo "Restoring from $BACKUP"
cp "$BACKUP/cnn_gru_multitask_sam_${TAG_OLD}_best.pt" pose_detection_live/best_model.pt
cp "$BACKUP/csi_normalization_stats_${TAG_OLD}.npz" outputs/
echo "GUI checkpoint -> pose_detection_live/best_model.pt"
echo "Norm stats     -> outputs/csi_normalization_stats_${TAG_OLD}.npz"
echo "To use in training/live, set EXPERIMENT_TAG=${TAG_OLD} in src/config.py"
