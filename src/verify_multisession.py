"""
Verify that every paired raw CSI/video session has corresponding
frames and pose labels, and optionally inspect the final NPZ.

Supports day1/day2/day3 subfolders under data/raw/csi and videos.
"""

from pathlib import Path
import numpy as np

from src.raw_paths import (
    iter_raw_csi,
    iter_raw_videos,
    session_id_from_raw_path,
)

ROOT = Path(__file__).resolve().parents[1]

FRAMES_ROOT = ROOT / "data" / "interim" / "frames"
LABEL_ROOT = ROOT / "data" / "interim" / "pose_labels"
NPZ_PATH = ROOT / "data" / "processed" / "csi_pose_dataset.npz"


def main():
    csi_by_session = {
        session_id_from_raw_path(p): p
        for p in iter_raw_csi()
    }
    video_by_session = {
        session_id_from_raw_path(p): p
        for p in iter_raw_videos()
    }

    paired = sorted(
        set(csi_by_session) & set(video_by_session)
    )

    print("RAW PAIRED SESSIONS")
    print("===================")
    print("count:", len(paired))

    missing_frames = []
    missing_labels = []

    for s in paired:
        frame_dir = FRAMES_ROOT / s
        label_file = LABEL_ROOT / f"{s}.json"
        is_empty = "empty_room" in s.lower()

        has_frames = (
            frame_dir.is_dir()
            and any(frame_dir.glob("*.png"))
        )
        has_label = label_file.exists()

        if not has_frames:
            missing_frames.append(s)
        if not is_empty and not has_label:
            missing_labels.append(s)

        csi_rel = csi_by_session[s].relative_to(ROOT / "data" / "raw" / "csi")
        print(
            f"{s:40s} "
            f"csi={csi_rel} "
            f"frames={'OK' if has_frames else 'MISSING':7s} "
            f"labels={'OK' if has_label or is_empty else 'MISSING'}"
        )

    print()
    print("Missing frames:", len(missing_frames))
    for s in missing_frames:
        print("  ", s)

    print()
    print("Missing pose labels:", len(missing_labels))
    for s in missing_labels:
        print("  ", s)

    only_csi = sorted(set(csi_by_session) - set(video_by_session))
    only_video = sorted(set(video_by_session) - set(csi_by_session))
    if only_csi:
        print("\nCSI without matching video:", len(only_csi))
        for s in only_csi:
            print("  ", s)
    if only_video:
        print("\nVideo without matching CSI:", len(only_video))
        for s in only_video:
            print("  ", s)

    if NPZ_PATH.exists():
        print()
        print("PROCESSED NPZ")
        print("=============")

        d = np.load(NPZ_PATH)
        sessions = d["session_id"].astype(str)
        subjects = d["subject_id"].astype(str)
        poses = d["pose_name"].astype(str)

        for sid in sorted(np.unique(subjects)):
            mask = subjects == sid
            print(f"\nSUBJECT {sid}")
            print("  samples:", int(mask.sum()))
            print(
                "  poses:",
                sorted(np.unique(poses[mask]).tolist()),
            )
            print(
                "  sessions:",
                sorted(np.unique(sessions[mask]).tolist())[:12],
            )


if __name__ == "__main__":
    main()
