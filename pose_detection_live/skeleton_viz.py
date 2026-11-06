"""Skeleton drawing helpers for live pose visualization."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO-style body connections (same as csi_pose_complete_project/test.py)
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]

# Colors for matplotlib rendering
EDGE_COLOR = "#00c9a7"
JOINT_COLOR = "#4a9eff"
HEAD_COLOR = "#ff7043"
BG_COLOR = "#0d1117"


def clip_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Clamp normalized coordinates to [0, 1]."""
    out = np.asarray(keypoints, dtype=np.float32).copy()
    np.clip(out, 0.0, 1.0, out=out)
    return out


def to_canvas_coords(
    keypoints: np.ndarray,
    width: int,
    height: int,
    margin: float = 0.08,
) -> np.ndarray:
    """Map normalized (x, y) keypoints to pixel coordinates with margin."""
    kp = clip_keypoints(keypoints)
    pad_x = width * margin
    pad_y = height * margin
    usable_w = width - 2 * pad_x
    usable_h = height - 2 * pad_y
    xs = pad_x + kp[:, 0] * usable_w
    ys = pad_y + kp[:, 1] * usable_h
    return np.stack([xs, ys], axis=1)


def draw_skeleton_on_axes(ax, keypoints: np.ndarray, title: str = "Live pose") -> None:
    """Draw skeleton on a matplotlib axis."""
    kp = clip_keypoints(keypoints)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(title, color="#dde1f0", fontsize=11)
    ax.tick_params(colors="#5a6080")
    for spine in ax.spines.values():
        spine.set_color("#252d40")

    for a, b in SKELETON_EDGES:
        ax.plot(
            [kp[a, 0], kp[b, 0]],
            [kp[a, 1], kp[b, 1]],
            color=EDGE_COLOR,
            linewidth=2.5,
            solid_capstyle="round",
            zorder=1,
        )

    head_idx = [0, 1, 2, 3, 4]
    body_idx = [i for i in range(len(kp)) if i not in head_idx]
    ax.scatter(kp[head_idx, 0], kp[head_idx, 1], c=HEAD_COLOR, s=55, zorder=3, edgecolors="white", linewidths=0.5)
    ax.scatter(kp[body_idx, 0], kp[body_idx, 1], c=JOINT_COLOR, s=45, zorder=3, edgecolors="white", linewidths=0.5)


def draw_skeleton_on_canvas(
    canvas,
    keypoints: np.ndarray,
    width: int,
    height: int,
    line_width: int = 3,
    joint_radius: int = 5,
) -> None:
    """Draw skeleton on a tkinter Canvas widget."""
    canvas.delete("all")
    canvas.configure(bg=BG_COLOR, highlightthickness=0)
    pts = to_canvas_coords(keypoints, width, height)

    for a, b in SKELETON_EDGES:
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        canvas.create_line(x1, y1, x2, y2, fill=EDGE_COLOR, width=line_width, capstyle="round")

    head_idx = {0, 1, 2, 3, 4}
    for i, (x, y) in enumerate(pts):
        color = HEAD_COLOR if i in head_idx else JOINT_COLOR
        r = joint_radius + (1 if i in head_idx else 0)
        canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=1)


def pose_summary(keypoints: np.ndarray, pose_name: str | None = None, pose_probs=None) -> str:
    """Short text summary for status display."""
    kp = clip_keypoints(keypoints)
    xs = kp[:, 0]
    ys = kp[:, 1]
    base = (
        f"joints=17  "
        f"x=[{xs.min():.2f}, {xs.max():.2f}]  "
        f"y=[{ys.min():.2f}, {ys.max():.2f}]"
    )
    if pose_name:
        conf = ""
        if pose_probs is not None:
            conf = f" ({float(np.max(pose_probs)):.0%})"
        return f"pose={pose_name}{conf}  {base}"
    return base
