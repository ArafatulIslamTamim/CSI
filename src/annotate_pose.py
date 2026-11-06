"""
Batch YOLO-Pose annotation for the WiFi CSI pose project.

This version fixes multi-position / multi-repetition session handling.

It scans:
    data/interim/frames/<SESSION_NAME>/

and writes:
    data/interim/pose_labels/<SESSION_NAME>.json

The SESSION_NAME is preserved exactly, for example:
    002_sitting_center_front
    002_sitting_center_front_2
    002_sitting_center_front_3
    002_sitting_far_front
    002_standing_far_front_3

Run from the project root:
    python -m src.annotate_pose

Useful options:
    python -m src.annotate_pose --force
    python -m src.annotate_pose --session 002_sitting_center_front_2
    python -m src.annotate_pose --no-visuals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAMES_ROOT = PROJECT_ROOT / "data" / "interim" / "frames"
POSE_LABELS_ROOT = PROJECT_ROOT / "data" / "interim" / "pose_labels"
POSE_VIS_ROOT = PROJECT_ROOT / "data" / "interim" / "pose_visualizations"

LOCAL_MODEL = PROJECT_ROOT / "models" / "yolo26n-pose.pt"
MODEL_SOURCE = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "yolo26n-pose.pt"


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate every extracted-frame session with YOLO-Pose."
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help=(
            "Process only one exact session name, e.g. "
            "002_sitting_center_front_2"
        ),
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Only annotate sessions from one day folder, e.g. day4",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite pose-label JSON files that already exist.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Do not save YOLO skeleton visualization images.",
    )
    return parser.parse_args()


def discover_session_dirs(session_name=None, day=None):
    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(
            f"Frame root does not exist: {FRAMES_ROOT}\n"
            "Run frame extraction first."
        )

    if session_name is not None:
        path = FRAMES_ROOT / session_name
        if not path.is_dir():
            raise FileNotFoundError(
                f"Requested session frame folder not found: {path}"
            )
        return [path]

    day = day.strip().lower() if day else None

    sessions = []
    for path in sorted(FRAMES_ROOT.iterdir()):
        if not path.is_dir():
            continue
        # Empty-room frames are not used as YOLO pose GT.
        if "empty_room" in path.name.lower():
            continue
        if day is not None and not path.name.startswith(f"{day}_"):
            continue
        if not any(path.glob("*.png")):
            continue
        sessions.append(path)

    return sessions


def select_person(result):
    """
    Select one target person.

    Current acquisition assumes one target subject. If YOLO returns more than
    one person, use the detection with the highest bounding-box confidence.
    """
    if result.keypoints is None:
        return None, None

    try:
        n_people = len(result.keypoints)
    except TypeError:
        n_people = 0

    if n_people == 0:
        return None, None

    if (
        result.boxes is not None
        and result.boxes.conf is not None
        and len(result.boxes.conf) > 0
    ):
        person_index = int(result.boxes.conf.argmax().item())
        person_confidence = float(
            result.boxes.conf[person_index].item()
        )
    else:
        person_index = 0
        person_confidence = None

    # Defensive guard if boxes/keypoints counts differ.
    person_index = min(person_index, n_people - 1)

    return person_index, person_confidence


def make_annotation(result, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    person_index, person_confidence = select_person(result)

    if person_index is None:
        return {
            "frame": image_path.name,
            "detected": False,
            "person_confidence": None,
            "frame_width": int(width),
            "frame_height": int(height),
            "keypoints": [],
        }

    xy = (
        result.keypoints.xy[person_index]
        .detach()
        .cpu()
        .numpy()
    )

    # Prefer Ultralytics normalized coordinates when available.
    if result.keypoints.xyn is not None:
        xyn = (
            result.keypoints.xyn[person_index]
            .detach()
            .cpu()
            .numpy()
        )
    else:
        xyn = xy.copy()
        xyn[:, 0] /= max(float(width), 1.0)
        xyn[:, 1] /= max(float(height), 1.0)

    if result.keypoints.conf is not None:
        confidence = (
            result.keypoints.conf[person_index]
            .detach()
            .cpu()
            .numpy()
        )
    else:
        confidence = [None] * len(xy)

    keypoints = []

    # COCO-Pose should give 17 joints. Keep a clear failure if a different
    # model/checkpoint is accidentally used.
    if len(xy) != len(KEYPOINT_NAMES):
        raise RuntimeError(
            f"{image_path}: expected 17 COCO keypoints, got {len(xy)}. "
            f"Check the YOLO pose checkpoint."
        )

    for i, name in enumerate(KEYPOINT_NAMES):
        conf_i = confidence[i]

        keypoints.append(
            {
                "id": i,
                "name": name,
                "x": float(xy[i][0]),
                "y": float(xy[i][1]),
                "x_normalized": float(xyn[i][0]),
                "y_normalized": float(xyn[i][1]),
                "confidence": (
                    None
                    if conf_i is None
                    else float(conf_i)
                ),
            }
        )

    return {
        "frame": image_path.name,
        "detected": True,
        "person_confidence": person_confidence,
        "frame_width": int(width),
        "frame_height": int(height),
        "keypoints": keypoints,
    }


def annotate_session(
    model,
    session_dir,
    force=False,
    save_visuals=True,
):
    session_name = session_dir.name

    # IMPORTANT:
    # Preserve the complete directory stem. Do not reduce
    # "002_sitting_center_front_2" to "002_sitting_front".
    json_path = POSE_LABELS_ROOT / f"{session_name}.json"
    vis_dir = POSE_VIS_ROOT / session_name

    if json_path.exists() and not force:
        print(
            f"[skip] {session_name}: already exists -> {json_path.name}"
        )
        return "skipped", 0, 0

    images = sorted(session_dir.glob("*.png"))

    if not images:
        print(f"[skip] {session_name}: no PNG frames")
        return "empty", 0, 0

    if save_visuals:
        vis_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    detected_count = 0

    print()
    print("=" * 72)
    print(f"SESSION: {session_name}")
    print(f"frames : {len(images)}")
    print(f"output : {json_path}")
    print("=" * 72)

    for i, image_path in enumerate(images, start=1):
        results = model(
            str(image_path),
            verbose=False,
        )
        result = results[0]

        annotation = make_annotation(
            result,
            image_path,
        )
        annotations.append(annotation)

        if annotation["detected"]:
            detected_count += 1

        if save_visuals:
            annotated_image = result.plot()
            cv2.imwrite(
                str(vis_dir / image_path.name),
                annotated_image,
            )

        if i == 1 or i % 50 == 0 or i == len(images):
            print(
                f"  {i:4d}/{len(images)} | "
                f"detected={detected_count}"
            )

    POSE_LABELS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        # Keep the same top-level list format used by the previous pipeline.
        json.dump(
            annotations,
            f,
            indent=2,
        )

    failed = len(images) - detected_count

    print(
        f"[done] {session_name}: "
        f"detected={detected_count}, "
        f"failed={failed}"
    )

    return "done", detected_count, failed


def main():
    args = parse_args()

    POSE_LABELS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.no_visuals:
        POSE_VIS_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    session_dirs = discover_session_dirs(
        session_name=args.session,
        day=args.day,
    )

    if not session_dirs:
        raise FileNotFoundError(
            f"No extracted-frame session folders found under {FRAMES_ROOT}"
            + (f" for {args.day}" if args.day else "")
        )

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Frames root  : {FRAMES_ROOT}")
    print(f"Labels root  : {POSE_LABELS_ROOT}")
    print(f"YOLO model   : {MODEL_SOURCE}")
    print(f"Sessions     : {len(session_dirs)}"
          + (f" ({args.day} only)" if args.day else ""))
    print()

    # Print exactly what will be processed. This makes it easy to confirm
    # that center/far/repetition names are not being collapsed.
    for session_dir in session_dirs:
        print(f"  - {session_dir.name}")

    print()
    model = YOLO(MODEL_SOURCE)

    done = 0
    skipped = 0
    total_detected = 0
    total_failed = 0

    for session_dir in session_dirs:
        status, detected, failed = annotate_session(
            model=model,
            session_dir=session_dir,
            force=args.force,
            save_visuals=not args.no_visuals,
        )

        if status == "done":
            done += 1
            total_detected += detected
            total_failed += failed
        elif status == "skipped":
            skipped += 1

    print()
    print("=" * 72)
    print("ANNOTATION COMPLETE")
    print("=" * 72)
    print(f"processed sessions : {done}")
    print(f"skipped sessions   : {skipped}")
    print(f"detected frames    : {total_detected}")
    print(f"failed frames      : {total_failed}")
    print(f"labels directory   : {POSE_LABELS_ROOT}")
    print()
    print("Next:")
    print("  python -m src.build_dataset")
    print()
    print(
        "Then verify that csi_pose_dataset.npz contains the full "
        "center/far/repetition session names."
    )


if __name__ == "__main__":
    main()
