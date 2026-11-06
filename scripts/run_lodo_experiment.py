"""
Leave-one-day-out (LODO) model selection on Days 1-5.

Usage:
  python scripts/run_lodo_experiment.py --variant amp9_med3
  python scripts/run_lodo_experiment.py --variant amp3_nomed --days day1 day2
  python scripts/run_lodo_experiment.py --final --epochs 31
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ALL_DAYS = ["day1", "day2", "day3", "day4", "day5"]

VARIANTS = {
    "amp9_med3": {
        "CSI_INPUT_CHANNELS": 9,
        "CSI_MEDIAN_FILTER_KERNEL": 3,
        "EXPERIMENT_TAG": "lodo_v2_9ch_med3",
    },
    "amp3_nomed": {
        "CSI_INPUT_CHANNELS": 3,
        "CSI_MEDIAN_FILTER_KERNEL": 0,
        "EXPERIMENT_TAG": "lodo_v2_3ch_nomed",
    },
}


def run(cmd, env):
    print("\n" + "=" * 70)
    print("RUN:", " ".join(cmd))
    print("=" * 70, flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=PROJECT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="amp9_med3",
        choices=sorted(VARIANTS),
    )
    parser.add_argument("--days", nargs="*", default=ALL_DAYS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Train on all five days with SPLIT_MODE=final_all_days",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    base = {
        "MODEL_NAME": "tcn_three_head_v2",
        "SPLIT_MODE": "loso_day",
        "INCLUDE_EMPTY_ROOM_IN_DATASET": True,
        "WINDOW_MODE": "time_causal",
        "TIME_WINDOW_DURATION_S": 0.6,
        "TIME_WINDOW_SNAPSHOTS": 32,
        "TIME_WINDOW_STRIDE_S": 0.3,
        "USE_SESSION_BALANCED_SAMPLER": True,
        "POSE_REPRESENTATION": "root_relative_v2",
        "CSI_FEATURE_MODE": "log_amp",
        "CSI_LOCAL_NORMALIZATION": "none",
        "EXCLUDED_SESSIONS": ["day5_004_standing_center_front"],
    }
    base.update(VARIANTS[args.variant])

    if args.final:
        overrides = dict(base)
        overrides["SPLIT_MODE"] = "final_all_days"
        overrides["EXPERIMENT_TAG"] = (
            overrides["EXPERIMENT_TAG"] + "_final_all5"
        )
        if args.epochs:
            overrides["NUM_EPOCHS"] = args.epochs
            overrides["FINAL_TRAIN_EPOCHS"] = args.epochs
        days = ["final"]
    else:
        days = args.days

    env_base = os.environ.copy()
    best_epochs = []

    for day in days:
        overrides = dict(base)
        if day != "final":
            overrides["LODO_HOLDOUT_DAY"] = day
            overrides["EXPERIMENT_TAG"] = (
                f"{base['EXPERIMENT_TAG']}_holdout_{day}"
            )
        if args.epochs:
            overrides["NUM_EPOCHS"] = args.epochs

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            json.dump(overrides, f, indent=2)
            override_path = f.name

        env = env_base.copy()
        env["CSI_CONFIG_OVERRIDES"] = override_path

        try:
            if not args.skip_build:
                run([sys.executable, "-m", "src.build_dataset"], env)

            if not args.skip_train:
                run([sys.executable, "-m", "src.train"], env)

            hist_path = PROJECT / "outputs" / (
                f"tcn_three_head_v2_{overrides['EXPERIMENT_TAG']}"
                "_training_history.json"
            )
            if hist_path.exists():
                hist = json.loads(hist_path.read_text())
                for row in hist:
                    if "val" in row:
                        pass
                # best epoch recorded in checkpoint name during train
        finally:
            Path(override_path).unlink(missing_ok=True)

    print("\nLODO complete. Compare holdout-day metrics before final training.")
    print("Then: python scripts/run_lodo_experiment.py --final --epochs <median>")


if __name__ == "__main__":
    main()
