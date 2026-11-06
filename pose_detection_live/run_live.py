#!/usr/bin/env python3
"""CLI runner for live CSI pose detection (cnn_gru_multitask_sam, no GUI)."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

from csi_preprocessor import LiveCSIBuffer
from pose_engine import DEFAULT_CHECKPOINT, PoseEngine
from skeleton_viz import pose_summary


def replay_json(engine: PoseEngine, json_path: Path, max_predictions: int = 20) -> None:
    with json_path.open("r") as f:
        data = json.load(f)
    records = data.get("records", [])
    if not records:
        raise SystemExit(f"No records in {json_path}")

    buffer = LiveCSIBuffer(csi_norm=engine.csi_norm)
    buffer.ingest_json_records(records)
    windows = buffer.iter_burst_inputs()
    if not windows:
        print(
            "Not enough CSI for a full window "
            "(need ~5 bursts of RX0/1/2 snapshots).",
            file=sys.stderr,
        )
        sys.exit(1)

    from collections import Counter

    votes = Counter()
    for i, sample in enumerate(windows[:max_predictions]):
        pred = engine.predict(sample)
        votes[pred["pose_name"]] += 1
        print(
            f"[{i + 1}] "
            f"{pose_summary(pred['keypoints'], pred['pose_name'], pred['pose_probs'])}"
        )
    print(f"majority={votes.most_common(1)[0][0]}  counts={dict(votes)}")


def run_live(
    engine: PoseEngine,
    host: str,
    user: str,
    password: str,
    wifi_if: str,
    port: int,
    duration_s: float,
) -> None:
    from bpi_ssh_worker import BPISshWorker

    data_q: queue.Queue = queue.Queue()
    status_q: queue.Queue = queue.Queue()
    worker = BPISshWorker(
        host=host,
        username=user,
        password=password,
        wifi_if=wifi_if,
        ssh_port=port,
        data_queue=data_q,
        status_queue=status_q,
    )
    worker.start()

    buffer = LiveCSIBuffer(csi_norm=engine.csi_norm)
    t0 = time.time()
    preds = 0

    print(f"Collecting CSI from {host} for {duration_s:.0f}s...")
    while time.time() - t0 < duration_s:
        while True:
            try:
                level, msg = status_q.get_nowait()
                print(f"[{level}] {msg}")
            except queue.Empty:
                break

        try:
            batch = data_q.get(timeout=0.2)
            buffer.ingest_worker_batch(batch)
        except queue.Empty:
            continue

        if buffer.ready:
            sample = buffer.latest_input()
            if sample is not None:
                pred = engine.predict(sample)
                preds += 1
                print(
                    f"[pred {preds}] "
                    f"{pose_summary(pred['keypoints'], pred['pose_name'], pred['pose_probs'])}"
                )

    worker.stop()
    print(f"Done. predictions={preds} snapshots={len(buffer.snapshots)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSI live pose detection CLI (cnn_gru_multitask_sam)"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--replay", type=Path, help="Replay CSI JSON instead of live BPI stream")
    parser.add_argument("--host", default="192.168.1.219")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="")
    parser.add_argument("--wifi-if", default="wifi0")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-predictions", type=int, default=20)
    args = parser.parse_args()

    engine = PoseEngine(args.checkpoint)
    info = engine.info()
    print(
        f"Loaded {info.get('model_name')} epoch={info['epoch']} "
        f"device={info['device']} csi_norm={info.get('csi_norm_file')}"
    )

    if args.replay:
        replay_json(engine, args.replay, args.max_predictions)
    else:
        run_live(
            engine,
            host=args.host,
            user=args.user,
            password=args.password,
            wifi_if=args.wifi_if,
            port=args.port,
            duration_s=args.duration,
        )


if __name__ == "__main__":
    main()
