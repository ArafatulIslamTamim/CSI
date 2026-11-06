import csv
from pathlib import Path

import cv2

from . import config
from .raw_paths import iter_raw_videos, session_id_from_raw_path


def extract_video(video_path: Path):
    session = session_id_from_raw_path(video_path)

    out_dir = (
        config.FRAME_DIR
        / session
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {video_path}"
        )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if source_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid FPS for {video_path}"
        )

    target_fps = min(
        config.TARGET_VIDEO_FPS,
        source_fps,
    )

    target_period = 1.0 / target_fps
    next_time = 0.0

    source_idx = 0
    output_idx = 0
    rows = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        t = (
            float(
                cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )
            )
            / 1000.0
        )

        if t <= 0 and source_idx > 0:
            t = source_idx / source_fps

        if (
            t + 0.5 / source_fps
            >= next_time
        ):
            filename = (
                f"frame_{output_idx:06d}.png"
            )

            frame_path = (
                out_dir / filename
            )

            if not cv2.imwrite(
                str(frame_path),
                frame,
            ):
                raise RuntimeError(
                    f"Could not write {frame_path}"
                )

            rows.append(
                {
                    "frame_idx":
                        output_idx,
                    "source_frame_idx":
                        source_idx,
                    "filename":
                        filename,
                    "relative_timestamp_s":
                        f"{t:.9f}",
                    "width":
                        width,
                    "height":
                        height,
                    "source_fps":
                        f"{source_fps:.6f}",
                    "target_fps":
                        f"{target_fps:.6f}",
                }
            )

            output_idx += 1
            next_time = (
                output_idx
                * target_period
            )

        source_idx += 1

    cap.release()

    csv_path = (
        out_dir
        / "frame_timestamps.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "source_frame_idx",
                "filename",
                "relative_timestamp_s",
                "width",
                "height",
                "source_fps",
                "target_fps",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[frames] {video_path.name}: "
        f"{output_idx} frames at "
        f"{target_fps:.2f} FPS"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract video frames at TARGET_VIDEO_FPS."
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Only process one day folder, e.g. day4",
    )
    args = parser.parse_args()

    config.ensure_directories()

    videos = iter_raw_videos()
    if args.day:
        day = args.day.strip().lower()
        videos = [v for v in videos if v.parent.name == day]

    if not videos:
        raise FileNotFoundError(
            f"No videos under {config.RAW_VIDEO_DIR}"
            + (f" for {args.day}" if args.day else "")
            + " (checked day1/day2/day3/day4 subfolders)"
        )

    print(f"[frames] extracting {len(videos)} videos"
          + (f" ({args.day} only)" if args.day else ""))

    for video in videos:
        print(
            f"[frames] {video.relative_to(config.RAW_VIDEO_DIR)} "
            f"-> session {session_id_from_raw_path(video)}"
        )
        extract_video(video)

    print("[frames] done")


if __name__ == "__main__":
    main()
