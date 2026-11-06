"""
P0 diagnostics: quantify YOLO-Pose ground-truth quality per pose and joint.

Reads the processed dataset (keypoint confidences from the YOLO-Pose
teacher), reports per-pose / per-joint confidence statistics, per-subject
bone-length consistency, and saves the lowest-confidence sitting frames
with the GT skeleton overlaid for visual spot-checking.

Usage:
    python scripts/analyze_gt_quality.py [--num-spot-checks 12]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.geometry import BONES  # noqa: E402

JOINT_NAMES = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
)

LOW_CONFIDENCE_THRESHOLDS = (0.3, 0.5)


def per_pose_confidence(confidence, poses):
    report = {}
    for pose in sorted(set(poses.tolist())):
        if pose == config.EMPTY_POSE_NAME:
            continue
        mask = poses == pose
        conf = confidence[mask]

        joints = {}
        for j, name in enumerate(JOINT_NAMES):
            entry = {
                "mean_confidence": float(conf[:, j].mean()),
                "std_confidence": float(conf[:, j].std()),
            }
            for threshold in LOW_CONFIDENCE_THRESHOLDS:
                entry[f"fraction_below_{threshold}"] = float(
                    (conf[:, j] < threshold).mean()
                )
            joints[name] = entry

        report[pose] = {
            "samples": int(mask.sum()),
            "mean_confidence_all_joints": float(conf.mean()),
            "per_joint": joints,
        }
    return report


def per_subject_bone_consistency(keypoints, confidence, poses, subjects):
    """
    Coefficient of variation of each GT bone length within one
    subject+pose. High values flag inconsistent/noisy YOLO labels
    (a real bone cannot change length).
    """
    report = {}
    for subject in sorted(set(subjects.tolist())):
        if subject == config.EMPTY_SUBJECT_ID:
            continue
        for pose in ("sitting", "standing"):
            mask = (subjects == subject) & (poses == pose)
            if mask.sum() < 10:
                continue
            kp = keypoints[mask]
            conf = confidence[mask]

            bone_cv = {}
            for a, b in BONES:
                lengths = np.linalg.norm(kp[:, a] - kp[:, b], axis=-1)
                reliable = (conf[:, a] > 0.5) & (conf[:, b] > 0.5)
                if reliable.sum() < 10:
                    continue
                lengths = lengths[reliable]
                mean = float(lengths.mean())
                if mean < 1e-6:
                    continue
                bone_cv[f"{JOINT_NAMES[a]}-{JOINT_NAMES[b]}"] = float(
                    lengths.std() / mean
                )

            report[f"{subject}_{pose}"] = {
                "samples": int(mask.sum()),
                "median_bone_length_cv": (
                    float(np.median(list(bone_cv.values())))
                    if bone_cv
                    else None
                ),
                "worst_bone": (
                    max(bone_cv, key=bone_cv.get) if bone_cv else None
                ),
                "worst_bone_cv": (
                    float(max(bone_cv.values())) if bone_cv else None
                ),
            }
    return report


def save_spot_checks(data, num_samples):
    import cv2

    from src.visualize import draw_pose

    poses = data["pose_name"].astype(str)
    confidence = data["confidence"].astype(np.float32)
    keypoints = data["keypoints"].astype(np.float32)
    frame_paths = data["frame_path"].astype(str)
    sessions = data["session_id"].astype(str)

    sitting = np.flatnonzero(poses == "sitting")
    # Rank sitting frames by mean lower-body confidence (worst first).
    lower_body = confidence[sitting][:, 11:].mean(axis=1)
    worst = sitting[np.argsort(lower_body)[:num_samples]]

    out_dir = config.QA_DIR / "gt_lowconf_sitting"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx in worst:
        image = cv2.imread(frame_paths[idx])
        if image is None:
            continue
        draw_pose(image, keypoints[idx], (0, 180, 0))
        cv2.putText(
            image,
            f"{sessions[idx]} | lower-body conf={confidence[idx, 11:].mean():.2f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        out_path = out_dir / f"lowconf_{int(idx):06d}.png"
        cv2.imwrite(str(out_path), image)
        saved += 1

    return saved, str(out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-spot-checks", type=int, default=12)
    args = parser.parse_args()

    config.ensure_directories()
    data = np.load(config.DATASET_FILE)

    poses = data["pose_name"].astype(str)
    subjects = data["subject_id"].astype(str)
    confidence = data["confidence"].astype(np.float32)
    keypoints = data["keypoints"].astype(np.float32)

    confidence_report = per_pose_confidence(confidence, poses)
    bone_report = per_subject_bone_consistency(
        keypoints, confidence, poses, subjects
    )

    saved, spot_dir = save_spot_checks(data, args.num_spot_checks)

    report = {
        "dataset": str(config.DATASET_FILE),
        "per_pose_confidence": confidence_report,
        "per_subject_bone_length_cv": bone_report,
        "spot_check_images": {"saved": saved, "directory": spot_dir},
    }

    out_path = config.QA_DIR / "gt_quality_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nGROUND-TRUTH QUALITY (YOLO-Pose teacher)")
    print("=" * 48)
    for pose, entry in confidence_report.items():
        print(
            f"\n{pose.upper()} ({entry['samples']} samples) "
            f"mean confidence {entry['mean_confidence_all_joints']:.3f}"
        )
        print(f"{'joint':12s} {'mean':>6s} {'<0.3':>7s} {'<0.5':>7s}")
        for name, joint in entry["per_joint"].items():
            print(
                f"{name:12s} {joint['mean_confidence']:6.3f} "
                f"{joint['fraction_below_0.3']:7.1%} "
                f"{joint['fraction_below_0.5']:7.1%}"
            )

    print("\nBONE-LENGTH CONSISTENCY (GT, coefficient of variation)")
    print(f"{'subject_pose':22s} {'median CV':>10s} {'worst bone':>24s}")
    for key, entry in bone_report.items():
        median_cv = entry["median_bone_length_cv"]
        print(
            f"{key:22s} "
            f"{median_cv if median_cv is None else f'{median_cv:10.3f}'} "
            f"{str(entry['worst_bone']):>24s} "
            f"(CV={entry['worst_bone_cv']:.3f})"
        )

    print(f"\nSpot-check images: {saved} saved to {spot_dir}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
