"""Tests for LODO v2 CSI features and config."""

import numpy as np

from src import config
from src.csi_features import build_feature_tensor, temporal_median_filter


def test_nine_channel_shape():
    t, rx, sc = 32, 3, 53
    z = np.ones((t, rx, sc), dtype=np.complex64)
    feat = build_feature_tensor(z, use_median=False, input_channels=9)
    assert feat.shape == (t, 9, sc)


def test_median_kernel3():
    x = np.arange(20, dtype=np.float32).reshape(20, 1, 1)
    y = temporal_median_filter(x, 3)
    assert y.shape == x.shape


def test_excluded_session_config():
    assert "day5_004_standing_center_front" in config.EXCLUDED_SESSIONS


def test_v2_model_registered():
    assert "tcn_three_head_v2" in config.VALID_MODELS
