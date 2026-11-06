"""
Live CSI -> pose inference.

Streams CSI records from a source, assembles bursts and 5-burst windows
exactly like training, and runs the trained model once per new burst
(~9 Hz). When the classifier says the room is empty, the skeleton is
suppressed.

Usage:

    python -m src.live.infer --file data/raw/csi/007_sitting_center_front.json
    python -m src.live.infer --file ... --speed 2 --show-video
    python -m src.live.infer --file ... --no-display

The replay source simulates the real-time stream from a recorded
capture; a real BPI transport only needs to implement CSISource.
"""

import argparse
import csv
import time
from collections import Counter, deque

import numpy as np
import torch

from .. import config
from ..losses import reconstruct_absolute_keypoints
from ..models import build_model, unpack_model_output
from ..raw_paths import session_id_from_raw_path
from ..visualize import draw_pose
from .pipeline import build_live_pipeline
from .sources import ReplaySource


class ClassSmoother:
    """Majority vote over the last N predicted classes."""

    def __init__(self, window):
        self._votes = deque(maxlen=window)

    def push(self, class_idx):
        self._votes.append(int(class_idx))
        return Counter(self._votes).most_common(1)[0][0]


class KeypointSmoother:
    """Exponential moving average over [17,2] keypoints."""

    def __init__(self, alpha):
        self.alpha = float(alpha)
        self._state = None

    def push(self, keypoints):
        if self._state is None:
            self._state = keypoints.copy()
        else:
            self._state = (
                self.alpha * keypoints
                + (1.0 - self.alpha) * self._state
            )
        return self._state

    def reset(self):
        self._state = None


class FrameLookup:
    """
    Optional replay extra: map a burst timestamp to the nearest extracted
    video frame of the same session, for a side-by-side sanity view.
    """

    def __init__(self, session, video_start_unix):
        self.frame_dir = config.FRAME_DIR / session
        timestamp_file = self.frame_dir / "frame_timestamps.csv"

        if not timestamp_file.exists():
            raise FileNotFoundError(
                f"--show-video needs extracted frames: {timestamp_file}"
            )

        with timestamp_file.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.filenames = [row["filename"] for row in rows]
        self.times = np.asarray(
            [
                float(video_start_unix)
                + float(row["relative_timestamp_s"])
                for row in rows
            ],
            dtype=np.float64,
        )

    def nearest(self, burst_ts):
        idx = int(np.argmin(np.abs(self.times - burst_ts)))
        return self.frame_dir / self.filenames[idx]


def load_model(device):
    if not config.BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Best checkpoint missing: {config.BEST_MODEL_FILE}. "
            "Train first."
        )

    model = build_model().to(device)
    checkpoint = torch.load(config.BEST_MODEL_FILE, map_location=device)

    if checkpoint.get("model_name") != config.MODEL_NAME:
        raise RuntimeError(
            "Checkpoint/model mismatch: checkpoint is "
            f"{checkpoint.get('model_name')!r}, config wants "
            f"{config.MODEL_NAME!r}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict(model, device, model_input):
    """One forward pass. Returns (keypoints [17,2], probs or None)."""
    x = torch.from_numpy(model_input).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(x)
        pred_sam, pose_logits = unpack_model_output(output)
        root = output.get("root") if isinstance(output, dict) else None
        keypoints = reconstruct_absolute_keypoints(pred_sam, root)[0].cpu().numpy()

        probs = None
        if pose_logits is not None:
            probs = torch.softmax(
                pose_logits[0],
                dim=0,
            ).cpu().numpy()

    return keypoints, probs


def render(canvas_shape, keypoints, label, stats_lines, frame=None):
    import cv2

    height, width = canvas_shape
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    if keypoints is not None:
        draw_pose(canvas, keypoints, (0, 0, 220))
    else:
        cv2.putText(
            canvas,
            "ROOM EMPTY",
            (width // 2 - 90, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for i, line in enumerate(stats_lines):
        cv2.putText(
            canvas,
            line,
            (10, height - 12 - 22 * (len(stats_lines) - 1 - i)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    if frame is not None:
        frame = cv2.resize(frame, (width, height))
        if keypoints is not None:
            draw_pose(frame, keypoints, (0, 0, 220))
        canvas = np.hstack([frame, canvas])

    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["replay"],
        default="replay",
        help="CSI transport (only replay is implemented so far)",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="recorded capture JSON to replay",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="replay speed factor (0 = as fast as possible)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="console output only",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="show the recorded video frame next to the prediction",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model, checkpoint = load_model(device)

    source = ReplaySource(args.file, speed=args.speed)
    pipeline = build_live_pipeline(
        idle_timeout_s=config.LIVE_BURST_IDLE_TIMEOUT_S
    )

    class_smoother = ClassSmoother(config.LIVE_CLASS_VOTE_WINDOW)
    keypoint_smoother = KeypointSmoother(config.LIVE_KEYPOINT_EMA_ALPHA)

    frame_lookup = None
    if args.show_video:
        # day3/007_standing_....json -> session day3_007_standing_....
        # (must match data/interim/frames/<session>/)
        session = session_id_from_raw_path(source.path)
        video_start = (
            source.metadata.get("video", {}).get("start_unix")
        )
        if video_start is None:
            raise RuntimeError(
                "--show-video needs metadata.video.start_unix "
                "in the replayed capture."
            )
        frame_lookup = FrameLookup(session, video_start)

    display = not args.no_display
    if display:
        import cv2

    print("\nLIVE INFERENCE")
    print("==============")
    print(f"model: {config.MODEL_NAME} (epoch {checkpoint.get('epoch')})")
    print(f"source: {args.source} {args.file} (speed x{args.speed})")
    print(f"classes: {config.POSE_CLASSES}")
    print(
        "empty gating: "
        f"P(empty) > {config.LIVE_EMPTY_PROB_THRESHOLD} "
        f"or vote over last {config.LIVE_CLASS_VOTE_WINDOW}"
    )
    print("press q in the window to quit\n")

    prediction_count = 0
    previous_burst_ts = None
    burst_rate = float("nan")

    def handle(item):
        nonlocal prediction_count, previous_burst_ts, burst_rate

        started = time.perf_counter()
        keypoints, probs = predict(model, device, item["input"])
        inference_ms = (time.perf_counter() - started) * 1000.0

        burst_ts = item["burst_ts"]
        if previous_burst_ts is not None and burst_ts > previous_burst_ts:
            step_rate = 1.0 / (burst_ts - previous_burst_ts)
            burst_rate = (
                step_rate
                if np.isnan(burst_rate)
                else 0.9 * burst_rate + 0.1 * step_rate
            )
        previous_burst_ts = burst_ts
        prediction_count += 1

        if probs is not None:
            raw_class = int(np.argmax(probs))
            voted_class = class_smoother.push(raw_class)
            if config.EMPTY_CLASS_INDEX >= 0:
                empty_prob = float(probs[config.EMPTY_CLASS_INDEX])
                is_empty = (
                    voted_class == config.EMPTY_CLASS_INDEX
                    or empty_prob > config.LIVE_EMPTY_PROB_THRESHOLD
                )
            else:
                empty_prob = float("nan")
                is_empty = False
            label = (
                f"{config.POSE_CLASSES[voted_class]} "
                f"({probs[voted_class]:.2f})"
            )
        else:
            # Single-task model: no classifier, no gating.
            is_empty = False
            empty_prob = float("nan")
            label = "pose (no classifier head)"

        if is_empty:
            keypoint_smoother.reset()
            shown_keypoints = None
        else:
            shown_keypoints = keypoint_smoother.push(keypoints)

        if display:
            stats = [
                f"inference {inference_ms:.1f} ms",
                f"burst rate {burst_rate:.1f} Hz",
                f"P(empty) {empty_prob:.2f}",
            ]

            frame = None
            if frame_lookup is not None:
                frame = cv2.imread(str(frame_lookup.nearest(burst_ts)))

            canvas = render(
                (config.LIVE_DISPLAY_HEIGHT, config.LIVE_DISPLAY_WIDTH),
                shown_keypoints,
                label,
                stats,
                frame=frame,
            )

            cv2.imshow("CSI live pose", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        else:
            print(
                f"[{prediction_count:05d}] "
                f"{label:24s} "
                f"P(empty)={empty_prob:.2f} "
                f"inference={inference_ms:.1f}ms "
                f"rate={burst_rate:.1f}Hz "
                f"{'(skeleton suppressed)' if is_empty else ''}"
            )

        return True

    try:
        for record in source:
            item = pipeline.poll()
            if item is not None and not handle(item):
                return

            item = pipeline.push_record(record)
            if item is not None and not handle(item):
                return

        item = pipeline.flush()
        if item is not None:
            handle(item)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if display:
            import cv2
            cv2.destroyAllWindows()

    print(f"\ndone: {prediction_count} predictions")


if __name__ == "__main__":
    main()
