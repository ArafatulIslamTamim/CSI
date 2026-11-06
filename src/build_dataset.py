import csv
import json
from pathlib import Path

import numpy as np

from . import config
from .csi_parser import (
    load_csi_json,
    group_complete_snapshots,
    group_bursts,
    amplitude_from_complex,
)
from .sam import generate_sam


def find_csi_file(
    session: str,
) -> Path:
    from .raw_paths import find_raw_csi

    csi_path = find_raw_csi(session)

    if csi_path is None:
        raise FileNotFoundError(
            f"No exact CSI JSON for session "
            f"'{session}' under {config.RAW_CSI_DIR} "
            "(checked day1/day2/day3 subfolders)"
        )

    return csi_path

def load_pose_labels(
    session: str,
):
    path = (
        config.POSE_LABEL_DIR
        / f"{session}.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    # Old/canonical format:
    #
    # {
    #     "session": ...,
    #     "labels": [...]
    # }
    if isinstance(obj, dict):
        labels = obj.get("labels")

        if not isinstance(labels, list):
            raise ValueError(
                f"{path}: expected 'labels' "
                "to be a list."
            )

        return labels

    # New YOLO annotation format:
    #
    # [
    #     {
    #         "frame": ...,
    #         "detected": ...,
    #         "frame_width": ...,
    #         "frame_height": ...,
    #         "keypoints": ...
    #     },
    #     ...
    # ]
    if isinstance(obj, list):
        timestamp_path = (
            config.FRAME_DIR
            / session
            / "frame_timestamps.csv"
        )

        if not timestamp_path.exists():
            raise FileNotFoundError(
                f"{path}: LIST-format pose labels "
                "require timestamp file: "
                f"{timestamp_path}"
            )

        with timestamp_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            timestamp_rows = list(
                csv.DictReader(f)
            )

        timestamp_by_filename = {
            row["filename"]: row
            for row in timestamp_rows
        }

        if len(obj) != len(timestamp_rows):
            raise ValueError(
                f"{session}: pose-label count "
                f"({len(obj)}) != timestamp count "
                f"({len(timestamp_rows)})"
            )

        normalized = []

        for annotation in obj:
            filename = annotation.get(
                "frame"
            )

            if not filename:
                raise ValueError(
                    f"{session}: LIST annotation "
                    "missing 'frame'."
                )

            row = timestamp_by_filename.get(
                filename
            )

            if row is None:
                raise ValueError(
                    f"{session}: frame "
                    f"'{filename}' has no matching "
                    "timestamp row."
                )

            frame_path = (
                config.FRAME_DIR
                / session
                / filename
            )

            normalized.append(
                {
                    "frame_idx":
                        int(row["frame_idx"]),

                    "filename":
                        filename,

                    "frame_path":
                        str(frame_path),

                    "relative_timestamp_s":
                        float(
                            row[
                                "relative_timestamp_s"
                            ]
                        ),

                    "width":
                        int(
                            annotation.get(
                                "frame_width",
                                row["width"],
                            )
                        ),

                    "height":
                        int(
                            annotation.get(
                                "frame_height",
                                row["height"],
                            )
                        ),

                    "was_inferred":
                        False,

                    "valid_pose":
                        bool(
                            annotation.get(
                                "detected",
                                False,
                            )
                        ),

                    "keypoints":
                        annotation.get(
                            "keypoints",
                            [],
                        ),
                }
            )

        return normalized

    raise ValueError(
        f"{path}: unsupported top-level "
        f"JSON type: {type(obj).__name__}"
    )


def infer_subject_id(
    session: str,
) -> str:
    """
    Person ID from a session name.

    Supports both:
      001_sitting_center_front
      day1_001_sitting_center_front
    """
    parts = session.split("_")
    if (
        len(parts) >= 2
        and parts[0] in {"day1", "day2", "day3", "day4", "day5"}
    ):
        return parts[1]
    return parts[0]


def infer_pose_name(
    metadata: dict,
    session: str,
) -> str:
    value = metadata.get("pose")

    if value:
        return str(value)

    low = session.lower()

    if "sitting" in low:
        return "sitting"

    if "standing" in low:
        return "standing"

    return "unknown"


def is_session_excluded(session: str) -> bool:
    excluded = getattr(config, "EXCLUDED_SESSIONS", ())
    return str(session) in excluded or any(
        str(session).endswith(s) for s in excluded
    )


def normalize_pose(
    label: dict,
):
    kp = label.get("keypoints")

    if (
        not label.get("valid_pose")
        or not isinstance(kp, list)
        or len(kp) != config.NUM_KEYPOINTS
    ):
        return None

    width = float(label["width"])
    height = float(label["height"])

    xy = np.asarray(
        [
            [
                float(p["x"]) / width,
                float(p["y"]) / height,
            ]
            for p in kp
        ],
        dtype=np.float32,
    )

    conf = np.asarray(
        [
            float(
                p.get(
                    "confidence",
                    1.0,
                )
            )
            for p in kp
        ],
        dtype=np.float32,
    )

    min_vis = int(getattr(config, "MIN_VISIBLE_KEYPOINTS", 0))
    min_conf = float(getattr(config, "MIN_KEYPOINT_CONFIDENCE", 0.0))
    if min_vis > 0:
        visible = int((conf >= min_conf).sum())
        if visible < min_vis:
            return None

    return xy, conf


def nearest_valid_frame(
    labels,
    frame_absolute_times,
    target_ts,
):
    """
    Return nearest frame that also has valid YOLO-Pose output.
    """
    valid_indices = [
        i
        for i, label in enumerate(labels)
        if (
            label.get("valid_pose")
            and label.get("keypoints") is not None
        )
    ]

    if not valid_indices:
        return None

    valid_times = np.asarray(
        [
            frame_absolute_times[i]
            for i in valid_indices
        ],
        dtype=np.float64,
    )

    local = int(
        np.argmin(
            np.abs(
                valid_times
                - target_ts
            )
        )
    )

    frame_index = valid_indices[local]
    error = float(
        abs(
            frame_absolute_times[
                frame_index
            ]
            - target_ts
        )
    )

    return frame_index, error


def build_empty_room_profile():
    if config.CSI_FEATURE_MODE != "empty_room_delta_db":
        return None

    from .raw_paths import iter_empty_room_csi

    candidates = iter_empty_room_csi()

    if len(candidates) != 1:
        raise FileNotFoundError(
            "CSI_FEATURE_MODE='empty_room_delta_db' "
            f"requires one {config.EMPTY_ROOM_CAPTURE_KEY}.json"
        )

    _, records = load_csi_json(
        candidates[0]
    )

    snapshots = group_complete_snapshots(
        records
    )

    if not snapshots:
        raise RuntimeError(
            "No valid empty-room snapshots."
        )

    amplitude = np.stack(
        [
            amplitude_from_complex(
                s.complex_csi
            )
            for s in snapshots
        ],
        axis=0,
    )

    profile = amplitude.mean(
        axis=0
    ).astype(np.float32)

    np.savez_compressed(
        config.EMPTY_ROOM_PROFILE_FILE,
        mean_amplitude=profile,
    )

    print(
        "[empty-room] saved profile: "
        f"{config.EMPTY_ROOM_PROFILE_FILE}"
    )

    return profile


def transform_feature(
    complex_window,
    empty_profile=None,
):
    if config.CSI_FEATURE_MODE in ("log_amp", "log_amp_9ch"):
        from .csi_features import build_feature_tensor

        return build_feature_tensor(complex_window)

    amplitude = amplitude_from_complex(
        complex_window
    )

    if config.CSI_FEATURE_MODE == "amplitude":
        return amplitude

    if (
        config.CSI_FEATURE_MODE
        == "empty_room_delta_db"
    ):
        if empty_profile is None:
            raise RuntimeError(
                "Empty-room profile missing."
            )

        return (
            20.0
            * np.log10(
                (
                    amplitude
                    + config.EMPTY_ROOM_EPS
                )
                / (
                    empty_profile[
                        None,
                        None,
                        :,
                        :
                    ]
                    + config.EMPTY_ROOM_EPS
                )
            )
        ).astype(np.float32)

    raise ValueError(
        "Unknown CSI_FEATURE_MODE="
        f"{config.CSI_FEATURE_MODE}"
    )


def process_empty_session(
    csi_path: Path,
    empty_profile,
):
    """
    Build "empty" class samples from an empty-room capture.

    No pose labels exist, so keypoints/confidence/SAM are all zeros:
    confidence 0 disables coordinate/SAM supervision, and the sample
    only trains the classification head.
    """
    from .raw_paths import session_id_from_raw_path

    session = session_id_from_raw_path(csi_path)

    metadata, records = load_csi_json(
        csi_path
    )

    video_meta = metadata.get("video", {})
    video_start = video_meta.get("start_unix")
    video_stop = video_meta.get("stop_unix")
    frame_width = int(video_meta.get("width", 0))
    frame_height = int(video_meta.get("height", 0))

    snapshots = group_complete_snapshots(records)
    bursts = group_bursts(snapshots)

    if not bursts:
        print(f"[WARN] {session}: no bursts.")
        return None

    zero_xy = np.zeros(
        (config.NUM_KEYPOINTS, 2),
        dtype=np.float32,
    )
    zero_conf = np.zeros(
        config.NUM_KEYPOINTS,
        dtype=np.float32,
    )
    zero_sam = np.zeros(
        (
            config.SAM_CHANNELS,
            config.NUM_KEYPOINTS,
            config.NUM_KEYPOINTS,
        ),
        dtype=np.float32,
    )

    samples = {
        "csi": [],
        "sam": [],
        "keypoints": [],
        "confidence": [],
        "session_id": [],
        "subject_id": [],
        "pose_name": [],
        "frame_path": [],
        "frame_width": [],
        "frame_height": [],
        "frame_idx": [],
        "frame_absolute_ts": [],
        "target_burst_ts": [],
        "sync_error_s": [],
        "burst_timestamps": [],
    }

    for burst_idx, target_burst in enumerate(bursts):
        target_ts = float(target_burst.center_ts)

        # Keep the same in-video restriction as pose sessions when the
        # capture provides video timing.
        if (
            video_start is not None
            and video_stop is not None
            and not (
                float(video_start)
                <= target_ts
                <= float(video_stop)
            )
        ):
            continue

        start = burst_idx - config.BURSTS_PER_WINDOW + 1
        if start < 0:
            continue

        window = bursts[start:burst_idx + 1]

        if len(window) != config.BURSTS_PER_WINDOW:
            continue

        span = window[-1].center_ts - window[0].center_ts
        if span > config.MAX_BURST_WINDOW_SPAN_S:
            continue

        complex_window = np.stack(
            [b.complex_csi for b in window],
            axis=0,
        )

        feature = transform_feature(
            complex_window,
            empty_profile=empty_profile,
        )

        samples["csi"].append(feature.astype(np.float32))
        samples["sam"].append(zero_sam.copy())
        samples["keypoints"].append(zero_xy.copy())
        samples["confidence"].append(zero_conf.copy())
        samples["session_id"].append(session)
        samples["subject_id"].append(config.EMPTY_SUBJECT_ID)
        samples["pose_name"].append(config.EMPTY_POSE_NAME)
        samples["frame_path"].append("")
        samples["frame_width"].append(frame_width)
        samples["frame_height"].append(frame_height)
        samples["frame_idx"].append(-1)
        samples["frame_absolute_ts"].append(target_ts)
        samples["target_burst_ts"].append(target_ts)
        samples["sync_error_s"].append(0.0)
        samples["burst_timestamps"].append(
            np.asarray(
                [b.center_ts for b in window],
                dtype=np.float64,
            )
        )

    if not samples["csi"]:
        print(f"[WARN] {session}: no empty-room windows.")
        return None

    result = {
        "csi": np.stack(samples["csi"]),
        "sam": np.stack(samples["sam"]),
        "keypoints": np.stack(samples["keypoints"]),
        "confidence": np.stack(samples["confidence"]),
        "session_id": np.asarray(samples["session_id"]),
        "subject_id": np.asarray(samples["subject_id"]),
        "pose_name": np.asarray(samples["pose_name"]),
        "frame_path": np.asarray(samples["frame_path"]),
        "frame_width": np.asarray(
            samples["frame_width"],
            dtype=np.int32,
        ),
        "frame_height": np.asarray(
            samples["frame_height"],
            dtype=np.int32,
        ),
        "frame_idx": np.asarray(
            samples["frame_idx"],
            dtype=np.int64,
        ),
        "frame_absolute_ts": np.asarray(
            samples["frame_absolute_ts"],
            dtype=np.float64,
        ),
        "target_burst_ts": np.asarray(
            samples["target_burst_ts"],
            dtype=np.float64,
        ),
        "sync_error_s": np.asarray(
            samples["sync_error_s"],
            dtype=np.float64,
        ),
        "burst_timestamps": np.stack(
            samples["burst_timestamps"]
        ),
    }

    print(
        f"[dataset] {session}: "
        f"{len(result['csi'])} empty-room samples"
    )

    return result


    return result


def process_session_time_causal(
    pose_path: Path,
    empty_profile,
):
    """Build samples using causal TIME_WINDOW_DURATION_S host_ts windows."""
    from .time_windows import (
        build_causal_time_window,
        iter_window_end_times,
    )

    session = pose_path.stem

    if is_session_excluded(session):
        print(f"[dataset] SKIP {session}: excluded session")
        return None

    csi_path = find_csi_file(session)

    metadata, records = load_csi_json(csi_path)

    video_meta = metadata.get("video", {})
    video_start = video_meta.get("start_unix")
    video_stop = video_meta.get("stop_unix")

    if video_start is None or video_stop is None:
        raise ValueError(
            f"{csi_path}: metadata.video.start_unix "
            "and stop_unix are required."
        )

    video_start = float(video_start)
    video_stop = float(video_stop)

    labels = load_pose_labels(session)

    frame_absolute_times = np.asarray(
        [
            video_start + float(label["relative_timestamp_s"])
            for label in labels
        ],
        dtype=np.float64,
    )

    snapshots = group_complete_snapshots(records)

    if not snapshots:
        print(f"[WARN] {session}: no snapshots.")
        return None

    samples = {
        "csi": [],
        "sam": [],
        "keypoints": [],
        "confidence": [],
        "session_id": [],
        "subject_id": [],
        "pose_name": [],
        "frame_path": [],
        "frame_width": [],
        "frame_height": [],
        "frame_idx": [],
        "frame_absolute_ts": [],
        "target_burst_ts": [],
        "sync_error_s": [],
        "burst_timestamps": [],
    }

    pose_name = infer_pose_name(metadata, session)

    skipped_no_history = 0
    skipped_outside_video = 0
    skipped_pose = 0

    duration_s = float(config.TIME_WINDOW_DURATION_S)
    n_snapshots = int(config.TIME_WINDOW_SNAPSHOTS)
    stride_s = float(getattr(config, "TIME_WINDOW_STRIDE_S", duration_s))
    max_sync = float(
        getattr(config, "TIME_WINDOW_MAX_SYNC_ERROR_S", 0.100)
    )
    n_channels = int(getattr(config, "CSI_INPUT_CHANNELS", 3))
    expected = (
        n_snapshots,
        n_channels,
        config.NUM_USABLE_SUBCARRIERS,
    )

    end_times = iter_window_end_times(
        video_start,
        video_stop,
        duration_s,
        stride_s,
    )

    for end_ts in end_times:
        if not (video_start <= end_ts <= video_stop):
            skipped_outside_video += 1
            continue

        complex_window = build_causal_time_window(
            snapshots,
            float(end_ts),
            duration_s=duration_s,
            n_snapshots=n_snapshots,
            min_snapshots=int(
                getattr(config, "TIME_WINDOW_MIN_SNAPSHOTS", 3)
            ),
        )

        if complex_window is None:
            skipped_no_history += 1
            continue

        label_match = nearest_valid_frame(
            labels,
            frame_absolute_times,
            float(end_ts),
        )
        if label_match is None:
            skipped_pose += 1
            continue
        label_idx, sync_error = label_match
        if sync_error > max_sync:
            skipped_pose += 1
            continue

        label = labels[label_idx]

        if complex_window.shape != (
            n_snapshots,
            config.NUM_RX,
            config.NUM_USABLE_SUBCARRIERS,
        ):
            raise RuntimeError(
                f"{session}: CSI shape {complex_window.shape} "
                f"!= ({n_snapshots}, {config.NUM_RX}, 53)"
            )

        pose = normalize_pose(label)

        if pose is None:
            skipped_pose += 1
            continue

        xy, confidence = pose

        feature = transform_feature(
            complex_window,
            empty_profile=empty_profile,
        )

        if feature.shape != expected:
            raise RuntimeError(
                f"{session}: feature shape {feature.shape} != {expected}"
            )

        samples["csi"].append(feature.astype(np.float32))
        samples["sam"].append(generate_sam(xy).astype(np.float32))
        samples["keypoints"].append(xy.astype(np.float32))
        samples["confidence"].append(confidence.astype(np.float32))
        samples["session_id"].append(session)
        samples["subject_id"].append(infer_subject_id(session))
        samples["pose_name"].append(pose_name)
        samples["frame_path"].append(label["frame_path"])
        samples["frame_width"].append(int(label["width"]))
        samples["frame_height"].append(int(label["height"]))
        samples["frame_idx"].append(int(label["frame_idx"]))
        samples["frame_absolute_ts"].append(float(end_ts))
        samples["target_burst_ts"].append(float(end_ts))
        samples["sync_error_s"].append(float(sync_error))
        samples["burst_timestamps"].append(
            np.linspace(
                end_ts - duration_s,
                end_ts,
                n_snapshots,
                dtype=np.float64,
            )
        )

    if not samples["csi"]:
        print(f"[WARN] {session}: no synchronized samples.")
        return None

    result = {
        "csi": np.stack(samples["csi"]),
        "sam": np.stack(samples["sam"]),
        "keypoints": np.stack(samples["keypoints"]),
        "confidence": np.stack(samples["confidence"]),
        "session_id": np.asarray(samples["session_id"]),
        "subject_id": np.asarray(samples["subject_id"]),
        "pose_name": np.asarray(samples["pose_name"]),
        "frame_path": np.asarray(samples["frame_path"]),
        "frame_width": np.asarray(samples["frame_width"], dtype=np.int32),
        "frame_height": np.asarray(samples["frame_height"], dtype=np.int32),
        "frame_idx": np.asarray(samples["frame_idx"], dtype=np.int64),
        "frame_absolute_ts": np.asarray(
            samples["frame_absolute_ts"],
            dtype=np.float64,
        ),
        "target_burst_ts": np.asarray(
            samples["target_burst_ts"],
            dtype=np.float64,
        ),
        "sync_error_s": np.asarray(samples["sync_error_s"], dtype=np.float64),
        "burst_timestamps": np.stack(samples["burst_timestamps"]),
    }

    print(
        f"[dataset] {session}: "
        f"{len(result['csi'])} time-window samples | "
        f"skip outside={skipped_outside_video}, "
        f"history={skipped_no_history}, "
        f"pose={skipped_pose}"
    )

    return result


def process_session(
    pose_path: Path,
    empty_profile,
):
    if getattr(config, "WINDOW_MODE", "burst") == "time_causal":
        return process_session_time_causal(
            pose_path,
            empty_profile,
        )

    session = pose_path.stem

    csi_path = find_csi_file(
        session
    )

    metadata, records = load_csi_json(
        csi_path
    )

    video_meta = metadata.get(
        "video",
        {}
    )

    video_start = video_meta.get(
        "start_unix"
    )

    video_stop = video_meta.get(
        "stop_unix"
    )

    if (
        video_start is None
        or video_stop is None
    ):
        raise ValueError(
            f"{csi_path}: metadata.video.start_unix "
            "and stop_unix are required."
        )

    video_start = float(
        video_start
    )

    video_stop = float(
        video_stop
    )

    labels = load_pose_labels(
        session
    )

    frame_absolute_times = np.asarray(
        [
            video_start
            + float(
                label[
                    "relative_timestamp_s"
                ]
            )
            for label in labels
        ],
        dtype=np.float64,
    )

    snapshots = group_complete_snapshots(
        records
    )

    bursts = group_bursts(
        snapshots
    )

    if not bursts:
        print(
            f"[WARN] {session}: no bursts."
        )
        return None

    samples = {
        "csi": [],
        "sam": [],
        "keypoints": [],
        "confidence": [],
        "session_id": [],
        "subject_id": [],
        "pose_name": [],
        "frame_path": [],
        "frame_width": [],
        "frame_height": [],
        "frame_idx": [],
        "frame_absolute_ts": [],
        "target_burst_ts": [],
        "sync_error_s": [],
        "burst_timestamps": [],
    }

    pose_name = infer_pose_name(
        metadata,
        session,
    )

    skipped_no_history = 0
    skipped_outside_video = 0
    skipped_pose = 0
    skipped_sync = 0

    for burst_idx, target_burst in enumerate(
        bursts
    ):
        target_ts = float(
            target_burst.center_ts
        )

        if not (
            video_start
            <= target_ts
            <= video_stop
        ):
            skipped_outside_video += 1
            continue

        start = (
            burst_idx
            - config.BURSTS_PER_WINDOW
            + 1
        )

        if start < 0:
            skipped_no_history += 1
            continue

        window = bursts[
            start:
            burst_idx + 1
        ]

        if (
            len(window)
            != config.BURSTS_PER_WINDOW
        ):
            skipped_no_history += 1
            continue

        span = (
            window[-1].center_ts
            - window[0].center_ts
        )

        if (
            span
            > config.MAX_BURST_WINDOW_SPAN_S
        ):
            skipped_no_history += 1
            continue

        nearest = nearest_valid_frame(
            labels,
            frame_absolute_times,
            target_ts,
        )

        if nearest is None:
            skipped_pose += 1
            continue

        label_idx, sync_error = nearest

        if (
            sync_error
            > config.MAX_BURST_TO_FRAME_SYNC_ERROR_S
        ):
            skipped_sync += 1
            continue

        label = labels[
            label_idx
        ]

        pose = normalize_pose(
            label
        )

        if pose is None:
            skipped_pose += 1
            continue

        xy, confidence = pose

        complex_window = np.stack(
            [
                b.complex_csi
                for b in window
            ],
            axis=0,
        )
        # [K bursts, 5 snapshots, 3 Rx, 53]

        expected = (
            config.BURSTS_PER_WINDOW,
            config.SNAPSHOTS_PER_BURST,
            config.NUM_RX,
            config.NUM_USABLE_SUBCARRIERS,
        )

        if (
            complex_window.shape
            != expected
        ):
            raise RuntimeError(
                f"{session}: CSI shape "
                f"{complex_window.shape} "
                f"!= {expected}"
            )

        feature = transform_feature(
            complex_window,
            empty_profile=empty_profile,
        )

        samples["csi"].append(
            feature.astype(np.float32)
        )

        samples["sam"].append(
            generate_sam(
                xy
            ).astype(np.float32)
        )

        samples[
            "keypoints"
        ].append(
            xy.astype(np.float32)
        )

        samples[
            "confidence"
        ].append(
            confidence.astype(np.float32)
        )

        samples[
            "session_id"
        ].append(session)

        samples[
            "subject_id"
        ].append(
            infer_subject_id(
                session
            )
        )

        samples[
            "pose_name"
        ].append(pose_name)

        samples[
            "frame_path"
        ].append(
            label["frame_path"]
        )

        samples[
            "frame_width"
        ].append(
            int(label["width"])
        )

        samples[
            "frame_height"
        ].append(
            int(label["height"])
        )

        samples[
            "frame_idx"
        ].append(
            int(label["frame_idx"])
        )

        samples[
            "frame_absolute_ts"
        ].append(
            float(
                frame_absolute_times[
                    label_idx
                ]
            )
        )

        samples[
            "target_burst_ts"
        ].append(
            target_ts
        )

        samples[
            "sync_error_s"
        ].append(
            sync_error
        )

        samples[
            "burst_timestamps"
        ].append(
            np.asarray(
                [
                    b.center_ts
                    for b in window
                ],
                dtype=np.float64,
            )
        )

    if not samples["csi"]:
        print(
            f"[WARN] {session}: "
            "no synchronized samples."
        )
        return None

    result = {
        "csi":
            np.stack(samples["csi"]),
        "sam":
            np.stack(samples["sam"]),
        "keypoints":
            np.stack(samples["keypoints"]),
        "confidence":
            np.stack(samples["confidence"]),
        "session_id":
            np.asarray(samples["session_id"]),
        "subject_id":
            np.asarray(samples["subject_id"]),
        "pose_name":
            np.asarray(samples["pose_name"]),
        "frame_path":
            np.asarray(samples["frame_path"]),
        "frame_width":
            np.asarray(
                samples["frame_width"],
                dtype=np.int32,
            ),
        "frame_height":
            np.asarray(
                samples["frame_height"],
                dtype=np.int32,
            ),
        "frame_idx":
            np.asarray(
                samples["frame_idx"],
                dtype=np.int64,
            ),
        "frame_absolute_ts":
            np.asarray(
                samples["frame_absolute_ts"],
                dtype=np.float64,
            ),
        "target_burst_ts":
            np.asarray(
                samples["target_burst_ts"],
                dtype=np.float64,
            ),
        "sync_error_s":
            np.asarray(
                samples["sync_error_s"],
                dtype=np.float64,
            ),
        "burst_timestamps":
            np.stack(
                samples[
                    "burst_timestamps"
                ]
            ),
    }

    print(
        f"[dataset] {session}: "
        f"{len(result['csi'])} samples | "
        f"median sync error="
        f"{np.median(result['sync_error_s'])*1000:.2f} ms | "
        f"skip outside={skipped_outside_video}, "
        f"history={skipped_no_history}, "
        f"pose={skipped_pose}, "
        f"sync={skipped_sync}"
    )

    return result


def main():
    config.ensure_directories()

    empty_profile = (
        build_empty_room_profile()
    )

    pose_files = sorted(
        p
        for p in config.POSE_LABEL_DIR.glob(
            "*.json"
        )
        if (
            "empty_room"
            not in p.stem.lower()
        )
    )

    if not pose_files:
        raise FileNotFoundError(
            "No YOLO-Pose pose labels. "
            "Run python -m src.annotate_pose first."
        )

    pieces = []

    for pose_path in pose_files:
        session = pose_path.stem

        if is_session_excluded(session):
            print(f"[dataset] SKIP {session}: excluded session")
            continue

        from .raw_paths import find_raw_csi

        csi_path = find_raw_csi(session)

        if csi_path is None:
            print(
                f"[dataset] SKIP {session}: "
                "no exact matching CSI JSON"
            )
            continue

        result = process_session(
            pose_path,
            empty_profile,
        )

        if result is not None:
            pieces.append(result)

    if getattr(config, "INCLUDE_EMPTY_ROOM_IN_DATASET", True):
        # Empty-room captures produce "empty" class samples without pose labels.
        from .raw_paths import iter_empty_room_csi

        empty_csi_files = iter_empty_room_csi()

        if not empty_csi_files:
            print(
                "[dataset] WARN: no empty-room CSI captures found; "
                "the 'empty' class will be missing."
            )

        for csi_path in empty_csi_files:
            result = process_empty_session(
                csi_path,
                empty_profile,
            )

            if result is not None:
                pieces.append(result)
    else:
        print(
            "[dataset] INCLUDE_EMPTY_ROOM_IN_DATASET=False; "
            "skipping empty-room CSI sessions."
        )

    if not pieces:
        raise RuntimeError(
            "No dataset samples were created."
        )

    keys = pieces[0].keys()

    dataset = {
        key:
            np.concatenate(
                [
                    p[key]
                    for p in pieces
                ],
                axis=0,
            )
        for key in keys
    }

    if not np.isfinite(
        dataset["csi"]
    ).all():
        raise RuntimeError(
            "NaN/Inf in CSI tensor."
        )

    if not np.isfinite(
        dataset["sam"]
    ).all():
        raise RuntimeError(
            "NaN/Inf in SAM labels."
        )

    np.savez_compressed(
        config.DATASET_FILE,
        **dataset,
    )

    print(
        "\nDATASET CREATED"
        "\n==============="
    )

    print(
        f"samples: "
        f"{len(dataset['csi'])}"
    )

    print(
        f"CSI shape: "
        f"{dataset['csi'].shape}"
    )

    print(
        f"SAM shape: "
        f"{dataset['sam'].shape}"
    )

    print(
        f"subjects: "
        f"{sorted(np.unique(dataset['subject_id']).tolist())}"
    )

    print(
        f"poses: "
        f"{sorted(np.unique(dataset['pose_name']).tolist())}"
    )

    print(
        f"median sync error: "
        f"{np.median(dataset['sync_error_s'])*1000:.2f} ms"
    )

    print(
        f"saved: "
        f"{config.DATASET_FILE}"
    )


if __name__ == "__main__":
    main()
