#!/usr/bin/env python3
"""Compare live empty-room CSI vs Dataset empties using Pipeline 3 training prep."""

from __future__ import annotations

import json
import queue
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bpi_ssh_worker import BPISshWorker
from csi_preprocessor import (
    LiveCSIBuffer,
    SIGNAL_SUBCARRIER_INDICES,
    apply_gain_normalization,
    normalize_amplitude,
)
from pose_engine import PoseEngine

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "csi_video_collector" / "Dataset"
CHECKPOINT = ROOT / "best_model.pt"
OUT_JSON = ROOT / "debug_empty_room_compare_p3.json"

HOST = "192.168.1.222"
USER = "root"
PASSWORD = "1234"
PORT = 22
WIFI_IF = "wifi1"
DURATION_S = 20.0


def amp_stats(windows: list[np.ndarray]) -> dict:
    """windows: list of (T, RX, K) raw amplitude."""
    stack = np.stack(windows, axis=0).astype(np.float32)
    active = stack[..., list(SIGNAL_SUBCARRIER_INDICES)]
    rx_means = [float(active[:, :, rx, :].mean()) for rx in range(active.shape[2])]
    return {
        "n": int(stack.shape[0]),
        "mean": float(active.mean()),
        "std": float(active.std()),
        "median": float(np.median(active)),
        "p5": float(np.percentile(active, 5)),
        "p95": float(np.percentile(active, 95)),
        "rx": rx_means,
        "rx0": rx_means[0],
        "rx1": rx_means[1],
        "rx2": rx_means[2],
    }


def mean_spectrum(windows: list[np.ndarray], gain_norm: bool) -> np.ndarray:
    """Mean (RX, K) spectrum; optional per-window gain-norm first."""
    specs = []
    for w in windows:
        x = w.astype(np.float32)
        if gain_norm:
            x = apply_gain_normalization(x, mode="per_window_signal_mean")
        specs.append(x.mean(axis=0))  # (RX, K)
    return np.mean(np.stack(specs, axis=0), axis=0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    signal = list(SIGNAL_SUBCARRIER_INDICES)
    va = a[:, signal].ravel().astype(np.float64)
    vb = b[:, signal].ravel().astype(np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(va, vb) / (na * nb))


def spectral_centroid(spec_rx_k: np.ndarray) -> float:
    signal = np.asarray(SIGNAL_SUBCARRIER_INDICES, dtype=np.float64)
    vals = spec_rx_k[:, list(SIGNAL_SUBCARRIER_INDICES)].mean(axis=0).astype(np.float64)
    dens = vals.sum()
    if dens < 1e-12:
        return float("nan")
    return float((signal * vals).sum() / dens)


def predict_windows(engine: PoseEngine, windows: list[np.ndarray], csi_norm: dict) -> dict:
    votes = Counter()
    probs = []
    for w in windows:
        sample = normalize_amplitude(w, csi_norm)
        pred = engine.predict(sample)
        votes[pred["pose_name"]] += 1
        probs.append(pred["pose_probs"])
    avg = np.mean(np.stack(probs, axis=0), axis=0)
    return {
        "votes": dict(votes),
        "avg_probs": {
            "empty_room": float(avg[0]),
            "sitting": float(avg[1]),
            "standing": float(avg[2]),
        },
    }


def collect_live(duration_s: float) -> tuple[list[np.ndarray], dict]:
    data_q: queue.Queue = queue.Queue()
    status_q: queue.Queue = queue.Queue()
    worker = BPISshWorker(
        host=HOST,
        username=USER,
        password=PASSWORD,
        wifi_if=WIFI_IF,
        ssh_port=PORT,
        data_queue=data_q,
        status_queue=status_q,
    )
    worker.start()
    buffer = LiveCSIBuffer(csi_norm=None)  # raw windows; we normalize later
    t0 = time.time()
    print(f"Collecting live CSI from {HOST} ({WIFI_IF}) for {duration_s:.0f}s — room should be empty...")
    while time.time() - t0 < duration_s:
        while True:
            try:
                level, msg = status_q.get_nowait()
                print(f"  [{level}] {msg}")
            except queue.Empty:
                break
        try:
            batch = data_q.get(timeout=0.2)
            buffer.ingest_worker_batch(batch)
        except queue.Empty:
            continue
    worker.stop()

    # Drain remaining
    while True:
        try:
            batch = data_q.get_nowait()
            buffer.ingest_worker_batch(batch)
        except queue.Empty:
            break

    # Raw amplitude windows (no norm) from burst iterator
    raw_buf = LiveCSIBuffer(csi_norm=None)
    # Re-use snapshots already grouped
    windows = []
    # Build from buffer snapshots via iter on a temp with same snaps
    # LiveCSIBuffer.iter_burst_inputs applies normalize; with csi_norm=None it still z-scores.
    # So rebuild raw windows manually from ready bursts.
    from csi_preprocessor import CSI_PACKETS_PER_FRAME, build_amplitude_window, segment_bursts

    bursts = segment_bursts(list(buffer.snapshots))
    for burst in bursts:
        if len(burst) < CSI_PACKETS_PER_FRAME:
            continue
        windows.append(build_amplitude_window(burst[:CSI_PACKETS_PER_FRAME]))

    meta = {
        "duration_s": duration_s,
        "host": HOST,
        "wifi_if": WIFI_IF,
        "snapshots": len(buffer.snapshots),
        "windows": len(windows),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"  snapshots={meta['snapshots']} windows={meta['windows']}")
    return windows, meta


def load_train_empty_windows(path: Path, max_windows: int = 40) -> list[np.ndarray]:
    with path.open() as f:
        data = json.load(f)
    buffer = LiveCSIBuffer(csi_norm=None)
    buffer.ingest_json_records(data.get("records", []))
    from csi_preprocessor import CSI_PACKETS_PER_FRAME, build_amplitude_window, segment_bursts

    windows = []
    for burst in segment_bursts(list(buffer.snapshots)):
        if len(burst) < CSI_PACKETS_PER_FRAME:
            continue
        windows.append(build_amplitude_window(burst[:CSI_PACKETS_PER_FRAME]))
        if len(windows) >= max_windows:
            break
    return windows, data.get("metadata") or {}


def summarize_set(name: str, windows: list[np.ndarray], engine: PoseEngine) -> dict:
    if not windows:
        return {"name": name, "error": "no windows"}
    raw = amp_stats(windows)
    spec_raw = mean_spectrum(windows, gain_norm=False)
    spec_gn = mean_spectrum(windows, gain_norm=True)
    preds = predict_windows(engine, windows, engine.csi_norm)
    return {
        "name": name,
        "n_windows": len(windows),
        "raw_amp": raw,
        "spectral_centroid_raw": spectral_centroid(spec_raw),
        "spectral_centroid_gain_norm": spectral_centroid(spec_gn),
        "mean_spectrum_raw": spec_raw,
        "mean_spectrum_gain_norm": spec_gn,
        "pred_p3": preds,
    }


def main() -> None:
    engine = PoseEngine(CHECKPOINT)
    info = engine.info()
    print(
        f"P3 checkpoint epoch={info['epoch']} gain_norm={info['gain_norm_mode']} "
        f"empty_mean={'yes' if engine.csi_norm and engine.csi_norm.get('empty_mean') is not None else 'no'}"
    )

    live_windows, live_meta = collect_live(DURATION_S)
    if len(live_windows) < 5:
        raise SystemExit(f"Too few live windows ({len(live_windows)}). Check BPI CSI traffic.")

    live = summarize_set("live_empty", live_windows[:40], engine)

    train_files = sorted(DATASET.glob("000_empty_room*.json"))
    train_rows = []
    for path in train_files:
        windows, meta = load_train_empty_windows(path, max_windows=40)
        row = summarize_set(path.name, windows, engine)
        row["file"] = path.name
        row["created"] = meta.get("created")
        row["bpi_ip"] = meta.get("bpi_ip")
        row["wifi_if"] = meta.get("wifi_if")
        train_rows.append(row)
        print(
            f"train {path.name}: n={row['n_windows']} rawμ={row['raw_amp']['mean']:.1f} "
            f"pred={row['pred_p3']['votes']}"
        )

    # Primary reference: first empty (same BPI/wifi as live typically)
    ref = train_rows[0]
    live_spec_gn = live["mean_spectrum_gain_norm"]
    train_spec_gn = ref["mean_spectrum_gain_norm"]
    live_spec_raw = live["mean_spectrum_raw"]
    train_spec_raw = ref["mean_spectrum_raw"]

    # Also compare vs pooled train empties
    pooled_gn = np.mean(
        np.stack([r["mean_spectrum_gain_norm"] for r in train_rows], axis=0), axis=0
    ).astype(np.float32)
    pooled_raw = np.mean(
        np.stack([r["mean_spectrum_raw"] for r in train_rows], axis=0), axis=0
    ).astype(np.float32)

    comparison = {
        "vs_primary_train": {
            "train_file": ref["file"],
            "gain_live_over_train": live["raw_amp"]["mean"] / max(ref["raw_amp"]["mean"], 1e-6),
            "mean_diff_raw": live["raw_amp"]["mean"] - ref["raw_amp"]["mean"],
            "cosine_raw": cosine(live_spec_raw, train_spec_raw),
            "cosine_gain_norm": cosine(live_spec_gn, train_spec_gn),
            "centroid_raw_live": live["spectral_centroid_raw"],
            "centroid_raw_train": ref["spectral_centroid_raw"],
            "centroid_gn_live": live["spectral_centroid_gain_norm"],
            "centroid_gn_train": ref["spectral_centroid_gain_norm"],
            "rx_ratio_live_over_train": [
                live["raw_amp"]["rx"][i] / max(ref["raw_amp"]["rx"][i], 1e-6) for i in range(3)
            ],
        },
        "vs_pooled_train_empties": {
            "n_files": len(train_rows),
            "cosine_raw": cosine(live_spec_raw, pooled_raw),
            "cosine_gain_norm": cosine(live_spec_gn, pooled_gn),
            "gain_live_over_pooled": live["raw_amp"]["mean"]
            / max(float(np.mean([r["raw_amp"]["mean"] for r in train_rows])), 1e-6),
        },
        "pipeline3_note": (
            "Training uses per-window gain-norm then train-set mean/std; "
            "no fixed empty residual. Cosine on gain-norm spectrum is the "
            "primary shape match metric."
        ),
    }

    # Strip bulky arrays from saved JSON (keep scalars)
    def slim(row: dict) -> dict:
        out = {k: v for k, v in row.items() if k not in {"mean_spectrum_raw", "mean_spectrum_gain_norm"}}
        return out

    payload = {
        "assumption": "Room is empty during this live capture",
        "pipeline": "pipeline3",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_info": {
            "epoch": info["epoch"],
            "gain_norm_mode": info["gain_norm_mode"],
            "has_empty_mean": engine.csi_norm.get("empty_mean") is not None,
            "use_log1p": bool(engine.csi_norm.get("use_log1p")),
        },
        "live": {
            **live_meta,
            **slim(live),
        },
        "train_empties": [slim(r) for r in train_rows],
        "comparison": comparison,
    }

    # numpy → json
    def _default(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with OUT_JSON.open("w") as f:
        json.dump(payload, f, indent=2, default=_default)

    print("\n========== LIVE vs TRAIN (Pipeline 3 features) ==========")
    print(f"live windows: {live['n_windows']}  rawμ={live['raw_amp']['mean']:.1f}")
    print(f"train {ref['file']}: rawμ={ref['raw_amp']['mean']:.1f}")
    print(f"gain live/train: {comparison['vs_primary_train']['gain_live_over_train']:.4f}")
    print(f"cosine RAW:       {comparison['vs_primary_train']['cosine_raw']:.6f}")
    print(f"cosine GAIN-NORM: {comparison['vs_primary_train']['cosine_gain_norm']:.6f}")
    print(f"cosine GN pooled: {comparison['vs_pooled_train_empties']['cosine_gain_norm']:.6f}")
    print(f"live pred:  {live['pred_p3']}")
    print(f"train pred: {ref['pred_p3']}")
    print(f"saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
