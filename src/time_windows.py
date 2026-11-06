"""Causal time-based CSI windows (host_ts), resampled to fixed snapshot count."""

from __future__ import annotations

import numpy as np

from . import config
from .csi_parser import CompleteSnapshot


def snapshot_times_array(
    snapshots: list[CompleteSnapshot],
) -> np.ndarray:
    return np.asarray(
        [float(s.host_ts) for s in snapshots],
        dtype=np.float64,
    )


def _pool_in_window(
    snapshots: list[CompleteSnapshot],
    start_ts: float,
    end_ts: float,
) -> list[CompleteSnapshot]:
    return [
        s
        for s in snapshots
        if start_ts < float(s.host_ts) <= end_ts
    ]


def _max_gap_s(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    return float(np.max(np.diff(np.sort(times))))


def _interpolate_rx_subcarriers(
    pool: list[CompleteSnapshot],
    target_times: np.ndarray,
) -> np.ndarray | None:
    """
    Per-Rx linear interpolation of complex CSI onto target_times.
    Returns [T, 3, 53] complex64 or None if any Rx lacks coverage.
    """
    if not pool:
        return None

    times = snapshot_times_array(pool)
    order = np.argsort(times)
    times = times[order]
    pool_sorted = [pool[i] for i in order]

    t_min, t_max = float(times[0]), float(times[-1])
    if target_times[0] < t_min - 1e-3 or target_times[-1] > t_max + 1e-3:
        return None

    out = np.zeros(
        (len(target_times), config.NUM_RX, config.NUM_USABLE_SUBCARRIERS),
        dtype=np.complex64,
    )

    for rx in range(config.NUM_RX):
        rx_times = []
        rx_vals = []
        for snap in pool_sorted:
            rx_times.append(float(snap.host_ts))
            rx_vals.append(snap.complex_csi[rx])
        rx_times = np.asarray(rx_times, dtype=np.float64)
        rx_stack = np.stack(rx_vals, axis=0)

        for ti, t in enumerate(target_times):
            if t <= rx_times[0]:
                out[ti, rx] = rx_stack[0]
            elif t >= rx_times[-1]:
                out[ti, rx] = rx_stack[-1]
            else:
                j = int(np.searchsorted(rx_times, t, side="right") - 1)
                t0, t1 = rx_times[j], rx_times[j + 1]
                w = (t - t0) / max(t1 - t0, 1e-12)
                out[ti, rx] = (
                    (1.0 - w) * rx_stack[j] + w * rx_stack[j + 1]
                ).astype(np.complex64)

    return out


def validate_window_pool(
    pool: list[CompleteSnapshot],
    *,
    min_records_per_rx: int | None = None,
    max_internal_gap_s: float | None = None,
) -> bool:
    """Return False if the pool fails v2 window rejection rules."""
    min_records = int(
        min_records_per_rx
        if min_records_per_rx is not None
        else getattr(config, "TIME_WINDOW_MIN_RECORDS_PER_RX", 18)
    )
    max_gap = float(
        max_internal_gap_s
        if max_internal_gap_s is not None
        else getattr(config, "TIME_WINDOW_MAX_GAP_S", 0.10)
    )

    if len(pool) < min_records:
        return False

    times = snapshot_times_array(pool)
    if _max_gap_s(times) > max_gap:
        return False

    return True


def build_causal_time_window(
    snapshots: list[CompleteSnapshot],
    end_ts: float,
    duration_s: float | None = None,
    n_snapshots: int | None = None,
    min_snapshots: int = 3,
    *,
    use_linear_interp: bool | None = None,
) -> np.ndarray | None:
    """
    Select snapshots with host_ts in (end_ts - duration, end_ts] and
    resample to ``n_snapshots``.

    Returns complex array [T, 3, 53] or None if rejected.
    """
    duration_s = float(
        duration_s
        if duration_s is not None
        else getattr(config, "TIME_WINDOW_DURATION_S", 0.6)
    )
    n_snapshots = int(
        n_snapshots
        if n_snapshots is not None
        else getattr(config, "TIME_WINDOW_SNAPSHOTS", 32)
    )
    use_linear = bool(
        getattr(config, "TIME_WINDOW_LINEAR_INTERP", True)
        if use_linear_interp is None
        else use_linear_interp
    )

    start_ts = float(end_ts) - duration_s
    pool = _pool_in_window(snapshots, start_ts, float(end_ts))

    if use_linear:
        if not validate_window_pool(pool):
            return None
        target_times = np.linspace(
            start_ts,
            float(end_ts),
            n_snapshots,
            dtype=np.float64,
        )
        return _interpolate_rx_subcarriers(pool, target_times)

    if len(pool) < min_snapshots:
        return None

    target_times = np.linspace(
        start_ts,
        float(end_ts),
        n_snapshots,
        dtype=np.float64,
    )
    times = snapshot_times_array(pool)
    chosen = []
    for t in target_times:
        idx = int(np.argmin(np.abs(times - float(t))))
        chosen.append(pool[idx])

    return np.stack(
        [s.complex_csi for s in chosen],
        axis=0,
    )


def iter_window_end_times(
    video_start: float,
    video_stop: float,
    duration_s: float,
    stride_s: float,
) -> np.ndarray:
    """Training window end timestamps inside video overlap."""
    first_end = video_start + duration_s
    if first_end > video_stop:
        return np.asarray([], dtype=np.float64)
    return np.arange(
        first_end,
        video_stop + 1e-9,
        stride_s,
        dtype=np.float64,
    )
