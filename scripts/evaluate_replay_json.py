#!/usr/bin/env python3
"""Evaluate live/replay JSON inference (same pipeline as the GUI).

Compares replay predictions to:
  - pose class from the session filename (sitting / standing / empty)
  - nearest-frame YOLO keypoints when pose labels exist (PCK / MPJPE)

Usage:
  CSI_CONFIG_OVERRIDES=pose_detection_live/tcn_gui_overrides.json \\
  PYTHONPATH=. .venv/bin/python scripts/evaluate_replay_json.py \\
      --glob 'data/raw/csi/day4/005_*.json'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "pose_detection_live") not in sys.path:
    sys.path.insert(0, str(ROOT / "pose_detection_live"))

from src import config
from src.build_dataset import (
    infer_pose_name,
    load_pose_labels,
    nearest_valid_frame,
    normalize_pose,
)
from src.csi_parser import load_csi_json
from src.live.pipeline import build_live_pipeline
from src.metrics import mpjpe, normalized_pck, pixel_mpjpe, pixel_pck
from src.raw_paths import session_id_from_raw_path

from pose_engine import PoseEngine


def gt_pose_from_path(path: Path, metadata: dict) -> str:
    session = session_id_from_raw_path(path)
    pose = infer_pose_name(metadata, session)
    if pose == "unknown" and "empty" in path.stem.lower():
        return "empty_room"
    return pose


def load_label_times(session: str, metadata: dict):
    labels = load_pose_labels(session)
    video_start = (metadata.get("video") or {}).get("start_unix")
    if video_start is None:
        return labels, None
    times = np.asarray(
        [
            float(video_start) + float(label["relative_timestamp_s"])
            for label in labels
        ],
        dtype=np.float64,
    )
    return labels, times


def replay_windows(records: list[dict]) -> list[tuple[float, np.ndarray]]:
    pipeline = build_live_pipeline()
    out: list[tuple[float, np.ndarray]] = []
    for record in records:
        item = pipeline.push_record(record)
        if item is not None:
            out.append((float(item["burst_ts"]), item["input"]))
    return out


def evaluate_file(
    engine: PoseEngine,
    path: Path,
    *,
    max_windows: int | None,
    window_stride: int,
) -> dict:
    metadata, records = load_csi_json(path)
    session = session_id_from_raw_path(path)
    gt_pose = gt_pose_from_path(path, metadata)

    t0 = time.perf_counter()
    windows = replay_windows(records)
    if window_stride > 1:
        windows = windows[::window_stride]
    if max_windows is not None:
        windows = windows[:max_windows]

    labels, frame_times = None, None
    try:
        labels, frame_times = load_label_times(session, metadata)
    except FileNotFoundError:
        pass

    vote = Counter()
    pred_ids = []
    pred_names = []
    pck_hits = []
    mpjpe_vals = []
    pixel_pck_vals = []
    pixel_mpjpe_vals = []
    matched_kp = 0

    for burst_ts, sample in windows:
        pred = engine.predict(sample)
        pred_names.append(pred["pose_name"])
        pred_ids.append(int(pred["pose_id"]))
        vote[pred["pose_name"]] += 1

        if labels is None or frame_times is None or gt_pose not in config.POSE_CLASSES:
            continue

        match = nearest_valid_frame(labels, frame_times, burst_ts)
        if match is None:
            continue
        idx, sync_err = match
        if sync_err > 0.20:
            continue
        norm = normalize_pose(labels[idx])
        if norm is None:
            continue
        gt_xy, _conf = norm
        pred_xy = pred["keypoints"]
        matched_kp += 1

        pck, _ = normalized_pck(
            pred_xy[None, ...],
            gt_xy[None, ...],
            config.PCK_ALPHA,
        )
        pck_hits.append(pck)
        m, _ = mpjpe(pred_xy[None, ...], gt_xy[None, ...])
        mpjpe_vals.append(m)
        px_pck = pixel_pck(
            pred_xy[None, ...],
            gt_xy[None, ...],
            np.asarray([labels[idx]["width"]], dtype=np.float32),
            np.asarray([labels[idx]["height"]], dtype=np.float32),
            config.PIXEL_PCK_THRESHOLDS,
        )
        pixel_pck_vals.append(float(px_pck.get("PCK@20", float("nan"))))
        pixel_mpjpe_vals.append(
            pixel_mpjpe(
                pred_xy[None, ...],
                gt_xy[None, ...],
                np.asarray([labels[idx]["width"]], dtype=np.float32),
                np.asarray([labels[idx]["height"]], dtype=np.float32),
            )
        )

    majority = vote.most_common(1)[0][0] if vote else None
    gt_id = (
        config.POSE_CLASSES.index(gt_pose)
        if gt_pose in config.POSE_CLASSES
        else None
    )
    window_acc = (
        float(np.mean([p == gt_id for p in pred_ids]))
        if gt_id is not None and pred_ids
        else None
    )

    return {
        "file": str(path if path.is_absolute() else (ROOT / path).relative_to(ROOT)),
        "session": session,
        "gt_pose": gt_pose,
        "n_records": len(records),
        "n_windows": len(windows),
        "majority_pred": majority,
        "majority_correct": (
            majority == gt_pose if gt_pose in config.POSE_CLASSES else None
        ),
        "vote_counts": dict(vote),
        "window_accuracy": window_acc,
        "matched_keypoint_windows": matched_kp,
        "normalized_pck": (
            float(np.mean(pck_hits)) if pck_hits else None
        ),
        "normalized_mpjpe": (
            float(np.mean(mpjpe_vals)) if mpjpe_vals else None
        ),
        "pixel_pck_0.2": (
            float(np.mean(pixel_pck_vals)) if pixel_pck_vals else None
        ),
        "pixel_mpjpe": (
            float(np.mean(pixel_mpjpe_vals)) if pixel_mpjpe_vals else None
        ),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def summarize(rows: list[dict]) -> dict:
    cls_rows = [
        r
        for r in rows
        if r["gt_pose"] in config.POSE_CLASSES and r["majority_correct"] is not None
    ]
    by_pose: dict[str, list[dict]] = {}
    for r in cls_rows:
        by_pose.setdefault(r["gt_pose"], []).append(r)

    per_pose = {}
    for pose, group in by_pose.items():
        per_pose[pose] = {
            "sessions": len(group),
            "majority_accuracy": float(
                np.mean([g["majority_correct"] for g in group])
            ),
            "mean_window_accuracy": float(
                np.mean([g["window_accuracy"] for g in group if g["window_accuracy"] is not None])
            ),
            "mean_pixel_pck_0.2": _nanmean(
                [g["pixel_pck_0.2"] for g in group]
            ),
            "mean_pixel_mpjpe": _nanmean(
                [g["pixel_mpjpe"] for g in group]
            ),
        }

    pck_rows = [r for r in rows if r.get("pixel_pck_0.2") is not None]
    return {
        "n_files": len(rows),
        "n_classified_sessions": len(cls_rows),
        "majority_accuracy": (
            float(np.mean([r["majority_correct"] for r in cls_rows]))
            if cls_rows
            else None
        ),
        "mean_window_accuracy": (
            float(np.mean([r["window_accuracy"] for r in cls_rows if r["window_accuracy"] is not None]))
            if cls_rows
            else None
        ),
        "mean_pixel_pck_0.2": _nanmean([r["pixel_pck_0.2"] for r in pck_rows]),
        "mean_pixel_mpjpe": _nanmean([r["pixel_mpjpe"] for r in pck_rows]),
        "per_pose": per_pose,
    }


def _nanmean(values):
    arr = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(arr)) if arr else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="data/raw/csi/day4/005_*.json",
        help="Glob of CSI JSON files (relative to repo root)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/checkpoints/tcn_multitask_sam_day125_val3_test4_tcn_v1_best.pt",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap windows per file (default: all, like GUI majority vote)",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=10,
        help="Use every Nth window (default 10 ≈ GUI-like majority, ~5 min for day4/005)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/evaluation/replay_json_eval_tcn_day4_005.json",
    )
    args = parser.parse_args()

    paths = sorted(ROOT.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched {args.glob!r}")

    engine = PoseEngine(args.checkpoint)
    info = engine.info()
    print(
        f"Model: {info.get('model_name')} epoch={info['epoch']} "
        f"window_mode={config.WINDOW_MODE}"
    )
    print(f"Evaluating {len(paths)} files matching {args.glob}\n")

    rows = []
    for path in paths:
        row = evaluate_file(
            engine,
            path,
            max_windows=args.max_windows,
            window_stride=args.window_stride,
        )
        rows.append(row)
        maj = "OK" if row["majority_correct"] else "MISS"
        if row["majority_correct"] is None:
            maj = "n/a"
        print(
            f"{maj}  {path.name}: gt={row['gt_pose']} "
            f"majority={row['majority_pred']} "
            f"win_acc={row['window_accuracy']} "
            f"pck@20={row['pixel_pck_0.2']} "
            f"({row['n_windows']} windows, {row['elapsed_s']}s)",
            flush=True,
        )

    summary = summarize(rows)
    payload = {
        "model": info.get("model_name"),
        "checkpoint": str(args.checkpoint),
        "epoch": info.get("epoch"),
        "window_mode": config.WINDOW_MODE,
        "experiment_tag": config.EXPERIMENT_TAG,
        "glob": args.glob,
        "summary": summary,
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
