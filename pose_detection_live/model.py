"""Re-exports for the trained cnn_gru_multitask_sam live stack.

Architecture lives in ``src.models`` (CNN-GRU backbone + SAM + class heads).
This module only exposes class labels used by the GUI / CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config  # noqa: E402

POSE_CLASSES = tuple(config.POSE_CLASSES)  # sitting, standing, empty
EMPTY_CLASS_NAME = config.EMPTY_POSE_NAME
EMPTY_CLASS_ID = int(config.EMPTY_CLASS_INDEX)
NUM_KEYPOINTS = int(config.NUM_KEYPOINTS)
