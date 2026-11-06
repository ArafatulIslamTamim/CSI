"""
Streaming (incremental) version of the offline CSI parser.

The offline pipeline in src/csi_parser.py sees the whole capture at once;
here the same grouping rules are applied one record at a time:

    record -> StreamingSnapshotAssembler -> CompleteSnapshot
    snapshot -> StreamingBurstAssembler  -> Burst (closed on time gap)
    burst -> SlidingWindowBuilder        -> [K,5,3,53] complex window
    window -> LiveNormalizer             -> normalized model input

A burst can only be closed when the next snapshot reveals the time gap,
or when poll() sees no snapshot for LIVE_BURST_IDLE_TIMEOUT_S of wall
time (stream idle / end of stream).
"""

import time
from collections import deque

import numpy as np

from .. import config
from ..csi_parser import (
    CompleteSnapshot,
    _pad_or_trim_burst,
    is_valid_record,
    parse_record_iq,
)
from ..dataset import _apply_local_csi_normalization


class StreamingSnapshotAssembler:
    """Incremental version of group_complete_snapshots."""

    def __init__(self):
        self._current = {}

    def push(self, record):
        """Feed one raw record; return a CompleteSnapshot when Rx0/1/2 complete."""
        if not is_valid_record(record):
            return None

        rx = int(record["rx_ant"])

        # Duplicate Rx before completion resets the partial group
        # (same rule as the offline parser).
        if rx in self._current:
            self._current = {}

        self._current[rx] = record

        if not all(
            rx_id in self._current
            for rx_id in config.RX_INDICES
        ):
            return None

        ordered = [
            self._current[rx_id]
            for rx_id in config.RX_INDICES
        ]

        times = np.asarray(
            [float(r["host_ts"]) for r in ordered],
            dtype=np.float64,
        )

        complex_csi = np.stack(
            [parse_record_iq(r) for r in ordered],
            axis=0,
        )

        snapshot = CompleteSnapshot(
            host_ts=float(np.median(times)),
            complex_csi=complex_csi,
            raw_indices=tuple(
                int(r.get("index", -1)) for r in ordered
            ),
            rssi=np.asarray(
                [float(r.get("rssi", np.nan)) for r in ordered],
                dtype=np.float32,
            ),
            snr=np.asarray(
                [float(r.get("snr", np.nan)) for r in ordered],
                dtype=np.float32,
            ),
        )

        self._current = {}
        return snapshot


class StreamingBurstAssembler:
    """
    Incremental version of group_bursts.

    Snapshots accumulate into the pending group; a host_ts gap larger
    than BURST_GAP_THRESHOLD_S closes the group as one burst. Bursts
    outside MIN/MAX_SNAPSHOTS_PER_BURST are dropped, matching offline.
    """

    def __init__(self, idle_timeout_s=None):
        self._pending = []
        self._last_arrival_wall = None
        self.idle_timeout_s = idle_timeout_s

    def _finalize(self):
        group = self._pending
        self._pending = []

        if not (
            config.MIN_SNAPSHOTS_PER_BURST
            <= len(group)
            <= config.MAX_SNAPSHOTS_PER_BURST
        ):
            return None

        return _pad_or_trim_burst(group)

    def push(self, snapshot):
        """Feed one snapshot; return the burst it closed, if any."""
        closed = None

        if (
            self._pending
            and (
                snapshot.host_ts - self._pending[-1].host_ts
                > config.BURST_GAP_THRESHOLD_S
            )
        ):
            closed = self._finalize()

        self._pending.append(snapshot)
        self._last_arrival_wall = time.monotonic()

        return closed

    def poll(self):
        """
        Close the pending group if the stream has been idle for longer
        than the wall-clock timeout. Call periodically in live mode.
        """
        if (
            not self._pending
            or self.idle_timeout_s is None
            or self._last_arrival_wall is None
        ):
            return None

        idle = time.monotonic() - self._last_arrival_wall

        if idle < self.idle_timeout_s:
            return None

        return self._finalize()

    def flush(self):
        """Close whatever is pending (end of stream)."""
        if not self._pending:
            return None

        return self._finalize()


class SlidingWindowBuilder:
    """Keep the most recent BURSTS_PER_WINDOW bursts and emit windows."""

    def __init__(self):
        self._bursts = deque(maxlen=config.BURSTS_PER_WINDOW)

    def push(self, burst):
        """
        Feed one burst; return the [K,5,3,53] complex window ending at
        this burst, or None when history is short or too spread in time.
        """
        self._bursts.append(burst)

        if len(self._bursts) < config.BURSTS_PER_WINDOW:
            return None

        span = (
            self._bursts[-1].center_ts
            - self._bursts[0].center_ts
        )

        if span > config.MAX_BURST_WINDOW_SPAN_S:
            return None

        return np.stack(
            [b.complex_csi for b in self._bursts],
            axis=0,
        )


class LiveNormalizer:
    """
    Reproduce the exact training-time feature transform:

        complex CSI -> feature (amplitude / empty-room delta dB)
        -> local per-snapshot normalization
        -> training-only per-Rx/per-subcarrier z-score
    """

    def __init__(self):
        self.empty_profile = None

        if config.CSI_FEATURE_MODE == "empty_room_delta_db":
            if not config.EMPTY_ROOM_PROFILE_FILE.exists():
                raise FileNotFoundError(
                    "CSI_FEATURE_MODE='empty_room_delta_db' requires "
                    f"{config.EMPTY_ROOM_PROFILE_FILE}. "
                    "Run python -m src.build_dataset first."
                )

            self.empty_profile = np.load(
                config.EMPTY_ROOM_PROFILE_FILE
            )["mean_amplitude"].astype(np.float32)

        self.mean = None
        self.std = None

        if config.NORMALIZE_CSI:
            if not config.CSI_NORM_FILE.exists():
                raise FileNotFoundError(
                    "Normalization statistics missing: "
                    f"{config.CSI_NORM_FILE}. Train first."
                )

            stats = np.load(config.CSI_NORM_FILE)
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

            saved_mode = (
                str(stats["local_normalization"].item()).lower()
                if "local_normalization" in stats.files
                else "none"
            )
            current_mode = str(
                getattr(config, "CSI_LOCAL_NORMALIZATION", "none")
            ).lower()

            if saved_mode != current_mode:
                raise RuntimeError(
                    "CSI normalization mode mismatch: stats use "
                    f"{saved_mode!r}, config requests {current_mode!r}."
                )

    def __call__(self, complex_window):
        """[K,5,3,53] complex -> [K,5,3,53] float32 model input."""
        # Deferred import: build_dataset pulls in the SAM module too.
        from ..build_dataset import transform_feature

        feature = transform_feature(
            complex_window,
            empty_profile=self.empty_profile,
        )

        feature = _apply_local_csi_normalization(feature)

        if self.mean is not None:
            if feature.ndim == 3:
                feature = (
                    feature - self.mean
                ) / (
                    self.std + config.CSI_NORM_EPS
                )
            else:
                feature = (
                    feature - self.mean[None, None, :, :]
                ) / (
                    self.std[None, None, :, :]
                    + config.CSI_NORM_EPS
                )

        return feature.astype(np.float32)


class TimeCausalWindowBuilder:
    """Keep recent snapshots and emit resampled causal windows."""

    def __init__(self):
        self._snapshots = deque()
        self._history_s = float(config.TIME_WINDOW_DURATION_S) + 0.15

    def push(self, snapshot):
        from ..time_windows import build_causal_time_window

        self._snapshots.append(snapshot)
        self._trim(float(snapshot.host_ts))

        return build_causal_time_window(
            list(self._snapshots),
            float(snapshot.host_ts),
            min_snapshots=int(
                getattr(config, "TIME_WINDOW_MIN_SNAPSHOTS", 3)
            ),
        )

    def _trim(self, now_ts: float):
        while self._snapshots:
            if now_ts - float(self._snapshots[0].host_ts) <= self._history_s:
                break
            self._snapshots.popleft()


class TimeCausalLiveCSIPipeline:
    """record -> snapshot -> causal time window -> normalized input."""

    def __init__(self):
        self.snapshots = StreamingSnapshotAssembler()
        self.windows = TimeCausalWindowBuilder()
        self.normalizer = LiveNormalizer()

    def push_record(self, record):
        snapshot = self.snapshots.push(record)
        if snapshot is None:
            return None

        complex_window = self.windows.push(snapshot)
        if complex_window is None:
            return None

        return {
            "input": self.normalizer(complex_window),
            "burst_ts": float(snapshot.host_ts),
        }

    def poll(self):
        return None

    def flush(self):
        return None


def build_live_pipeline(idle_timeout_s=None):
    mode = str(getattr(config, "WINDOW_MODE", "burst")).lower()
    if mode == "time_causal":
        return TimeCausalLiveCSIPipeline()
    return LiveCSIPipeline(idle_timeout_s=idle_timeout_s)


class LiveCSIPipeline:
    """record -> ... -> normalized model input, one call per record."""

    def __init__(self, idle_timeout_s=None):
        self.snapshots = StreamingSnapshotAssembler()
        self.bursts = StreamingBurstAssembler(
            idle_timeout_s=idle_timeout_s
        )
        self.windows = SlidingWindowBuilder()
        self.normalizer = LiveNormalizer()

    def _consume_burst(self, burst):
        if burst is None:
            return None

        complex_window = self.windows.push(burst)

        if complex_window is None:
            return None

        return {
            "input": self.normalizer(complex_window),
            "burst_ts": float(burst.center_ts),
        }

    def push_record(self, record):
        """Feed one raw record; return a model-ready item or None."""
        snapshot = self.snapshots.push(record)

        if snapshot is None:
            return None

        burst = self.bursts.push(snapshot)
        return self._consume_burst(burst)

    def poll(self):
        """Idle-timeout check; may emit the item for a force-closed burst."""
        return self._consume_burst(self.bursts.poll())

    def flush(self):
        """End of stream; may emit the item for the final burst."""
        return self._consume_burst(self.bursts.flush())
