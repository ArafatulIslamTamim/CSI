#!/usr/bin/env bash
# Launch live GUI with the day-split TCN checkpoint (500ms, 32 snapshots).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/pose_detection_live"

CKPT_SRC="$ROOT/outputs/checkpoints/tcn_multitask_sam_day125_val3_test4_tcn_v1_best.pt"
NORM_SRC="$ROOT/outputs/csi_normalization_stats_day125_val3_test4_tcn_v1.npz"

if [[ ! -f "$CKPT_SRC" ]]; then
  echo "Missing TCN checkpoint: $CKPT_SRC"
  echo "Train or restore checkpoint first."
  exit 1
fi
if [[ ! -f "$NORM_SRC" ]]; then
  echo "Missing norm stats: $NORM_SRC"
  echo "Run evaluation build (create_split + compute_and_save_csi_stats) first."
  exit 1
fi

cp "$CKPT_SRC" best_model.pt
export CSI_CONFIG_OVERRIDES="$ROOT/pose_detection_live/tcn_gui_overrides.json"
export PYTHONPATH="$ROOT"

echo "TCN GUI: checkpoint=$(basename "$CKPT_SRC") epoch metadata in best_model.pt"
echo "Norm stats: $NORM_SRC"
echo "Config overrides: $CSI_CONFIG_OVERRIDES"
exec "$ROOT/.venv/bin/python3" live_gui.py --checkpoint best_model.pt "$@"
