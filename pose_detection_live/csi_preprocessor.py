"""Live / replay CSI → normalized [5,5,3,53] windows (matches training).

Wraps ``src.live.pipeline.LiveCSIPipeline`` so the GUI/CLI keep the same
``LiveCSIBuffer`` API as before, but use the multitask SAM input layout.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional
from collections import deque

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config
from src.live.pipeline import build_live_pipeline, LiveCSIPipeline


def _complex_list_from_iq(I, Q) -> list[complex]:
    n = min(len(I), len(Q), config.RAW_NUM_SUBCARRIERS)
    out = [complex(float(I[i]), float(Q[i])) for i in range(n)]
    while len(out) < config.RAW_NUM_SUBCARRIERS:
        out.append(0j)
    return out[: config.RAW_NUM_SUBCARRIERS]


def worker_rec_to_training_record(rec, index: int = 0) -> Optional[dict]:
    """Convert BPI worker dict/list into a training-style CSI record."""
    from bpi_ssh_worker import list_record_to_dict

    if isinstance(rec, (list, tuple)):
        d = list_record_to_dict(rec)
    elif isinstance(rec, dict):
        d = list_record_to_dict(rec) if "I" in rec or "_raw" in rec else dict(rec)
    else:
        return None
    if d is None:
        return None

    # Already a formatted Dataset JSON record.
    if isinstance(d.get("iq"), list) and d.get("host_ts") is not None:
        out = dict(d)
        out.setdefault("tx_ant", config.TX_ANT)
        out.setdefault("bw", config.VALID_BW)
        out.setdefault("samples", config.VALID_SAMPLES)
        out.setdefault("mode", config.VALID_MODE)
        out.setdefault("ext", config.VALID_EXT)
        out.setdefault("index", index)
        return out

    I, Q = d.get("I"), d.get("Q")
    if not isinstance(I, list) or not isinstance(Q, list) or not I or not Q:
        return None

    raw = d.get("_raw")
    if isinstance(raw, (list, tuple)) and len(raw) >= 11:
        tx = int(raw[7])
        rx = int(raw[8])
    else:
        tx = int(d.get("tx_ant", config.TX_ANT))
        rx = int(d.get("rx_ant", 0))

    return {
        "index": index,
        "host_ts": float(d.get("_host_ts") or d.get("host_ts") or time.time()),
        "ts": int(raw[0]) if isinstance(raw, (list, tuple)) and raw else 0,
        "tx_ant": tx,
        "rx_ant": rx,
        "chain": int(d.get("chain", 0)),
        "bw": config.VALID_BW,
        "samples": config.VALID_SAMPLES,
        "mode": config.VALID_MODE,
        "ext": config.VALID_EXT,
        "rssi": float(d.get("rssi", np.nan)),
        "snr": float(d.get("snr", np.nan)),
        "iq": _complex_list_from_iq(I, Q),
    }


def record_from_json(rec: dict, index: int = 0) -> Optional[dict]:
    if not isinstance(rec, dict):
        return None
    out = dict(rec)
    if out.get("host_ts") is None and out.get("ts") is not None:
        out["host_ts"] = float(out["ts"])
    out.setdefault("index", index)
    out.setdefault("tx_ant", config.TX_ANT)
    out.setdefault("bw", config.VALID_BW)
    out.setdefault("samples", config.VALID_SAMPLES)
    out.setdefault("mode", config.VALID_MODE)
    out.setdefault("ext", config.VALID_EXT)
    if not isinstance(out.get("iq"), list):
        return None
    return out


@dataclass
class LiveCSIBuffer:
    """Rolling CSI → model windows via the training-matched live pipeline."""

    csi_norm: dict | None = None  # unused; LiveNormalizer reads project stats
    idle_timeout_s: float | None = None
    snapshots: Deque = field(default_factory=deque)
    _record_index: int = 0
    last_incomplete: int = 0
    last_spread_rejected: int = 0
    last_bursts_ready: int = 0
    _pipeline: LiveCSIPipeline | None = field(default=None, repr=False)
    _latest: np.ndarray | None = field(default=None, repr=False)
    _windows: list = field(default_factory=list, repr=False)
    _n_snapshots: int = 0

    def __post_init__(self) -> None:
        timeout = (
            self.idle_timeout_s
            if self.idle_timeout_s is not None
            else config.LIVE_BURST_IDLE_TIMEOUT_S
        )
        self._pipeline = build_live_pipeline(idle_timeout_s=timeout)
        self.snapshots = deque()  # length used as snapshot counter proxy

    def set_csi_norm(self, csi_norm: dict | None) -> None:
        self.csi_norm = csi_norm

    def clear(self) -> None:
        timeout = (
            self.idle_timeout_s
            if self.idle_timeout_s is not None
            else config.LIVE_BURST_IDLE_TIMEOUT_S
        )
        self._pipeline = build_live_pipeline(idle_timeout_s=timeout)
        self.snapshots.clear()
        self._record_index = 0
        self._latest = None
        self._windows = []
        self._n_snapshots = 0
        self.last_incomplete = 0
        self.last_spread_rejected = 0
        self.last_bursts_ready = 0

    def _emit(self, item: dict | None) -> None:
        if item is None:
            return
        arr = np.asarray(item["input"], dtype=np.float32)
        self._latest = arr
        self._windows.append(arr)
        self.last_bursts_ready = len(self._windows)

    def _push_record(self, record: dict) -> None:
        assert self._pipeline is not None
        item = self._pipeline.push_record(record)
        self._n_snapshots += 1
        self.snapshots.append(self._n_snapshots)
        while len(self.snapshots) > 400:
            self.snapshots.popleft()
        self._emit(item)

    def ingest_worker_batch(self, batch: list) -> int:
        added = 0
        for rec in batch:
            parsed = worker_rec_to_training_record(rec, self._record_index)
            if parsed is None:
                continue
            self._record_index += 1
            self._push_record(parsed)
            added += 1
        assert self._pipeline is not None
        self._emit(self._pipeline.poll())
        return added

    def ingest_json_records(self, records: list[dict]) -> int:
        added = 0
        for rec in records:
            parsed = record_from_json(rec, self._record_index)
            if parsed is None:
                continue
            self._record_index += 1
            self._push_record(parsed)
            added += 1
        assert self._pipeline is not None
        self._emit(self._pipeline.flush())
        return added

    @property
    def ready(self) -> bool:
        if self._latest is not None:
            return True
        assert self._pipeline is not None
        self._emit(self._pipeline.poll())
        return self._latest is not None

    def latest_input(self) -> Optional[np.ndarray]:
        assert self._pipeline is not None
        self._emit(self._pipeline.poll())
        return None if self._latest is None else self._latest.copy()

    def iter_burst_inputs(self) -> list[np.ndarray]:
        return [w.copy() for w in self._windows]

    def latest_raw_amplitude_window(self) -> Optional[np.ndarray]:
        return self.latest_input()

    def collect_raw_empty_mean(self, min_bursts: int = 8) -> np.ndarray:
        raise RuntimeError(
            "Empty residual calibration is not used by this model "
            f"(feature mode={config.CSI_FEATURE_MODE!r})."
        )

    def amp_stats(self) -> Optional[dict]:
        sample = self._latest
        if sample is None:
            return None
        return {
            "raw_mean": float(sample.mean()),
            "raw_std": float(sample.std()),
            "raw_max": float(sample.max()),
            "norm_mean": float(sample.mean()),
            "norm_std": float(sample.std()),
            "norm_min": float(sample.min()),
            "norm_max": float(sample.max()),
            "shape": tuple(sample.shape),
            "snapshots": len(self.snapshots),
            "bursts_ready": self.last_bursts_ready,
        }
