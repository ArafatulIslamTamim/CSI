"""
Resolve CSI / video files under optional day folders.

Layout:

    data/raw/csi/day1/<name>.json
    data/raw/videos/day1/<name>.mp4

Canonical session id becomes ``day1_<name>`` so frames, pose labels,
and CSI stay paired across days even when basenames repeat.
"""

from __future__ import annotations

from pathlib import Path

from . import config

DAY_FOLDER_NAMES = ("day1", "day2", "day3", "day4", "day5")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}


def session_id_from_raw_path(path: Path) -> str:
    """Map a raw CSI/video path to the flat session id used everywhere else."""
    stem = path.stem
    parent = path.parent.name
    if parent in DAY_FOLDER_NAMES:
        prefix = f"{parent}_"
        if stem.startswith(prefix):
            return stem
        return f"{parent}_{stem}"
    return stem


def is_empty_room_session(session_or_stem: str) -> bool:
    return "empty_room" in session_or_stem.lower()


def iter_raw_videos(root: Path | None = None) -> list[Path]:
    root = root or config.RAW_VIDEO_DIR
    if not root.exists():
        return []

    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: session_id_from_raw_path(p))


def iter_raw_csi(root: Path | None = None) -> list[Path]:
    root = root or config.RAW_CSI_DIR
    if not root.exists():
        return []

    files = [
        p
        for p in root.rglob("*.json")
        if p.is_file()
    ]
    return sorted(files, key=lambda p: session_id_from_raw_path(p))


def iter_empty_room_csi(root: Path | None = None) -> list[Path]:
    return [
        p
        for p in iter_raw_csi(root)
        if is_empty_room_session(p.stem)
        or is_empty_room_session(session_id_from_raw_path(p))
    ]


def find_raw_csi(session_id: str, root: Path | None = None) -> Path | None:
    """Locate the CSI JSON for a canonical session id."""
    root = root or config.RAW_CSI_DIR

    flat = root / f"{session_id}.json"
    if flat.is_file():
        return flat

    for day in DAY_FOLDER_NAMES:
        if not session_id.startswith(f"{day}_"):
            continue
        remainder = session_id[len(day) + 1 :]
        candidates = [
            root / day / f"{remainder}.json",
            root / day / f"{session_id}.json",
        ]
        for path in candidates:
            if path.is_file():
                return path

    for path in iter_raw_csi(root):
        if session_id_from_raw_path(path) == session_id:
            return path

    return None


def find_raw_video(session_id: str, root: Path | None = None) -> Path | None:
    root = root or config.RAW_VIDEO_DIR

    for ext in VIDEO_EXTENSIONS:
        flat = root / f"{session_id}{ext}"
        if flat.is_file():
            return flat

    for day in DAY_FOLDER_NAMES:
        if not session_id.startswith(f"{day}_"):
            continue
        remainder = session_id[len(day) + 1 :]
        for ext in VIDEO_EXTENSIONS:
            for path in (
                root / day / f"{remainder}{ext}",
                root / day / f"{session_id}{ext}",
            ):
                if path.is_file():
                    return path

    for path in iter_raw_videos(root):
        if session_id_from_raw_path(path) == session_id:
            return path

    return None
