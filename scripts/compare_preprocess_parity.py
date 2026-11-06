#!/usr/bin/env python3
"""
Compare offline dataset windows vs live pipeline on the same CSI JSON file.

Usage:
  PYTHONPATH=. python scripts/compare_preprocess_parity.py \
    --csi data/raw/csi/day5/005_standing_center_front.json \
    --session day5_005_standing_center_front
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.build_dataset import transform_feature
from src.csi_parser import group_bursts, group_complete_snapshots, load_csi_json
from src.dataset import _apply_local_csi_normalization
from src.live.pipeline import build_live_pipeline
from src.time_windows import build_causal_time_window


def _load_train_norm():
    if not config.CSI_NORM_FILE.exists():
        return None, None
    stats = np.load(config.CSI_NORM_FILE)
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def _apply_global_norm(feature, mean, std):
    if mean is None:
        return feature
    if feature.ndim == 3:
        return (feature - mean) / (std + config.CSI_NORM_EPS)
    return (feature - mean[None, None, :, :]) / (
        std[None, None, :, :] + config.CSI_NORM_EPS
    )


def offline_burst_windows(metadata, records):
    snapshots = group_complete_snapshots(records)
    bursts = group_bursts(snapshots)
    windows = []
    for burst_idx in range(config.BURSTS_PER_WINDOW - 1, len(bursts)):
        window_bursts = bursts[
            burst_idx - config.BURSTS_PER_WINDOW + 1 : burst_idx + 1
        ]
        if len(window_bursts) != config.BURSTS_PER_WINDOW:
            continue
        span = window_bursts[-1].center_ts - window_bursts[0].center_ts
        if span > config.MAX_BURST_WINDOW_SPAN_S:
            continue
        complex_window = np.stack(
            [b.complex_csi for b in window_bursts],
            axis=0,
        )
        feat = transform_feature(complex_window)
        feat = _apply_local_csi_normalization(feat)
        mean, std = _load_train_norm()
        feat = _apply_global_norm(feat, mean, std)
        windows.append(
            {
                "ts": float(window_bursts[-1].center_ts),
                "tensor": feat.astype(np.float32),
            }
        )
    return windows


def offline_from_npz(session: str, max_samples=500):
    """Ground-truth offline tensors from the built dataset."""
    if not config.DATASET_FILE.exists():
        return []
    data = np.load(config.DATASET_FILE)
    sessions = data["session_id"].astype(str)
    mask = sessions == session
    if not mask.any():
        return []
    csi = data["csi"][mask].astype(np.float32)
    ts = data["target_burst_ts"][mask].astype(np.float64)
    mean, std = _load_train_norm()
    windows = []
    for i in range(min(len(csi), max_samples)):
        feat = csi[i]
        # NPZ stores pre-global-norm features; apply train z-score like Dataset.
        feat = _apply_local_csi_normalization(feat)
        feat = _apply_global_norm(feat, mean, std)
        windows.append({"ts": float(ts[i]), "tensor": feat.astype(np.float32)})
    return windows


def offline_time_windows(metadata, records, session: str):
    """Build windows at pose label times (matches build_dataset)."""
    from src.build_dataset import load_pose_labels

    snapshots = group_complete_snapshots(records)
    video = metadata.get("video", {})
    start = float(video.get("start_unix", snapshots[0].host_ts))
    labels = load_pose_labels(session)
    duration = float(config.TIME_WINDOW_DURATION_S)
    windows = []
    for label in labels:
        end_ts = start + float(label["relative_timestamp_s"])
        complex_window = build_causal_time_window(snapshots, float(end_ts))
        if complex_window is None:
            continue
        feat = transform_feature(complex_window)
        feat = _apply_local_csi_normalization(feat)
        mean, std = _load_train_norm()
        feat = _apply_global_norm(feat, mean, std)
        windows.append({"ts": float(end_ts), "tensor": feat.astype(np.float32)})
    return windows


def live_windows(csi_path: Path):
    metadata, records = load_csi_json(csi_path)
    pipeline = build_live_pipeline()
    mean, std = _load_train_norm()
    windows = []
    for record in records:
        item = pipeline.push_record(record)
        if item is None:
            continue
        feat = np.asarray(item["input"], dtype=np.float32)
        # Live pipeline already normalizes; re-apply for fair compare if needed
        windows.append(
            {
                "ts": float(item["burst_ts"]),
                "tensor": feat,
            }
        )
    return windows


def match_and_diff(offline, live, max_pairs=200):
    if not offline or not live:
        return {
            "offline_count": len(offline),
            "live_count": len(live),
            "pairs": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }

    live_ts = np.asarray([w["ts"] for w in live], dtype=np.float64)
    max_diff = 0.0
    diffs = []
    pairs = 0

    for off in offline[:max_pairs]:
        idx = int(np.argmin(np.abs(live_ts - off["ts"])))
        if abs(live_ts[idx] - off["ts"]) > 0.05:
            continue
        a = off["tensor"]
        b = live[idx]["tensor"]
        if a.shape != b.shape:
            continue
        d = float(np.max(np.abs(a - b)))
        diffs.append(float(np.mean(np.abs(a - b))))
        max_diff = max(max_diff, d)
        pairs += 1

    return {
        "offline_count": len(offline),
        "live_count": len(live),
        "pairs": pairs,
        "max_abs_diff": max_diff if pairs else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi", type=Path, required=True)
    parser.add_argument("--session", type=str, default="")
    args = parser.parse_args()

    metadata, records = load_csi_json(args.csi)
    mode = str(getattr(config, "WINDOW_MODE", "burst")).lower()
    session = args.session or args.csi.stem.replace("_", "_", 1)
    # Infer session id from path: day5/005_standing... -> day5_005_standing...
    if not args.session:
        parts = args.csi.parts
        for i, p in enumerate(parts):
            if p.startswith("day") and i + 1 < len(parts):
                session = f"{p}_{Path(parts[-1]).stem}"
                break

    print(f"WINDOW_MODE={mode}")
    print(f"session={session}")
    print(f"CSI_NORM_FILE={config.CSI_NORM_FILE}")
    print("tensor shapes: burst [5,5,3,53] | time [32,3,53]")

    if mode == "time_causal":
        offline = offline_from_npz(session)
        if not offline:
            offline = offline_time_windows(metadata, records, session)
    else:
        offline = offline_burst_windows(metadata, records)

    live = live_windows(args.csi)
    report = match_and_diff(offline, live)

    print(json.dumps(report, indent=2))

    if report["pairs"] == 0:
        print("FAIL: no matched window pairs (check timing / WINDOW_MODE).")
        sys.exit(2)

    threshold = 1e-4
    if report["max_abs_diff"] is not None and report["max_abs_diff"] > threshold:
        print(
            f"WARN: max_abs_diff={report['max_abs_diff']:.6g} "
            f"> {threshold} — investigate preprocessing mismatch."
        )
        sys.exit(1)

    print(f"OK: max_abs_diff <= {threshold}")


if __name__ == "__main__":
    main()
