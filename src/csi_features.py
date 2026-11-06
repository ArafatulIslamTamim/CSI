"""CSI feature extraction: log amplitude, median filter, 9-channel stack."""

from __future__ import annotations

import numpy as np

from . import config
from .csi_parser import amplitude_from_complex


def log_amplitude(complex_csi: np.ndarray) -> np.ndarray:
    """complex [..., 3, 53] -> log amplitude same shape."""
    amp = amplitude_from_complex(complex_csi)
    eps = float(getattr(config, "LOG_AMP_EPS", 1e-9))
    return (20.0 * np.log10(amp + eps)).astype(np.float32)


def temporal_median_filter(
    x: np.ndarray,
    kernel: int = 3,
) -> np.ndarray:
    """
    Median filter along time axis (axis 0).
    x: [T, ...] — applied independently to every trailing dimension.
    """
    k = int(kernel)
    if k <= 1:
        return x.astype(np.float32, copy=False)

    t = x.shape[0]
    out = np.empty_like(x, dtype=np.float32)
    r = k // 2
    for i in range(t):
        sl = x[max(0, i - r) : min(t, i + r + 1)]
        out[i] = np.median(sl, axis=0)
    return out


def temporal_delta(x: np.ndarray) -> np.ndarray:
    """First temporal difference; t=0 set to zero."""
    d = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] > 1:
        d[1:] = x[1:] - x[:-1]
    return d


def rx_difference(x: np.ndarray) -> np.ndarray:
    """
    Cross-Rx differences on log-amplitude tensor [T, 3, S].
    Returns [T, 3, S] with channels (d01, d02, d12) broadcast as 3 Rx slots.
    """
    d01 = (x[:, 0] - x[:, 1]).astype(np.float32)
    d02 = (x[:, 0] - x[:, 2]).astype(np.float32)
    d12 = (x[:, 1] - x[:, 2]).astype(np.float32)
    return np.stack([d01, d02, d12], axis=1)


def build_feature_tensor(
    complex_window: np.ndarray,
    *,
    use_median: bool | None = None,
    input_channels: int | None = None,
) -> np.ndarray:
    """
    complex_window [T, 3, 53] -> feature [T, C, 53]
    C = 3 (amp only) or 9 (amp + delta + rx_diff).
    """
    use_median = (
        bool(getattr(config, "CSI_MEDIAN_FILTER_KERNEL", 0) > 1)
        if use_median is None
        else use_median
    )
    input_channels = int(
        input_channels
        if input_channels is not None
        else getattr(config, "CSI_INPUT_CHANNELS", 3)
    )

    log_a = log_amplitude(complex_window)
    if use_median:
        k = int(getattr(config, "CSI_MEDIAN_FILTER_KERNEL", 3))
        log_a = temporal_median_filter(log_a, k)

    if input_channels == 3:
        return log_a.astype(np.float32)

    if input_channels == 9:
        delta = temporal_delta(log_a)
        rx_diff = rx_difference(log_a)
        return np.concatenate(
            [log_a, delta, rx_diff],
            axis=1,
        ).astype(np.float32)

    raise ValueError(
        f"CSI_INPUT_CHANNELS must be 3 or 9, got {input_channels}"
    )


def clip_normalized(x: np.ndarray) -> np.ndarray:
    clip = float(getattr(config, "CSI_NORM_CLIP", 6.0))
    if clip <= 0:
        return x
    return np.clip(x, -clip, clip).astype(np.float32, copy=False)
