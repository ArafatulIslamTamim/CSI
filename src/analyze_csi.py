import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .csi_parser import (
    load_csi_json,
    group_complete_snapshots,
    group_bursts,
    estimate_snapshot_rate,
    estimate_burst_rate,
)


def analyze_file(path: Path):
    metadata, records = load_csi_json(path)

    snapshots = group_complete_snapshots(
        records
    )

    bursts = group_bursts(
        snapshots
    )

    valid_record_count = 0
    anomalies = {
        "bw_not_0": 0,
        "samples_not_64": 0,
        "mode_not_1": 0,
        "ext_not_0": 0,
        "iq_len_not_64": 0,
    }

    zero_counts = np.zeros(
        config.RAW_NUM_SUBCARRIERS,
        dtype=np.int64,
    )

    total_iq_records = 0

    for r in records:
        if int(r.get("bw", -1)) != config.VALID_BW:
            anomalies["bw_not_0"] += 1

        if int(r.get("samples", -1)) != config.VALID_SAMPLES:
            anomalies["samples_not_64"] += 1

        if int(r.get("mode", -1)) != config.VALID_MODE:
            anomalies["mode_not_1"] += 1

        if int(r.get("ext", -1)) != config.VALID_EXT:
            anomalies["ext_not_0"] += 1

        iq = r.get("iq")

        if isinstance(iq, list):
            if len(iq) != config.RAW_NUM_SUBCARRIERS:
                anomalies["iq_len_not_64"] += 1
            else:
                total_iq_records += 1

                for i, v in enumerate(iq):
                    try:
                        z = complex(
                            str(v)
                            .replace("+-", "-")
                        )
                    except Exception:
                        continue

                    if z == 0:
                        zero_counts[i] += 1

    if bursts:
        burst_sizes = []

        # Reconstruct raw groups for descriptive statistics.
        snapshots_sorted = sorted(
            snapshots,
            key=lambda s: s.host_ts,
        )

        group = [snapshots_sorted[0]]

        for s in snapshots_sorted[1:]:
            if (
                s.host_ts - group[-1].host_ts
                > config.BURST_GAP_THRESHOLD_S
            ):
                burst_sizes.append(len(group))
                group = [s]
            else:
                group.append(s)

        burst_sizes.append(len(group))

        median_burst_size = float(
            np.median(burst_sizes)
        )

        burst_centers = np.asarray(
            [b.center_ts for b in bursts],
            dtype=np.float64,
        )

        median_burst_gap = (
            float(np.median(np.diff(burst_centers)))
            if len(burst_centers) > 1
            else float("nan")
        )

    else:
        median_burst_size = float("nan")
        median_burst_gap = float("nan")

    video = metadata.get("video", {})

    result = {
        "file": path.name,
        "capture_key": metadata.get("capture_key"),
        "pose": metadata.get("pose"),
        "orientation": metadata.get("orientation"),
        "records_total": len(records),
        "complete_valid_snapshots": len(snapshots),
        "usable_bursts": len(bursts),
        "estimated_complete_snapshot_rate_hz":
            estimate_snapshot_rate(snapshots),
        "estimated_burst_rate_hz":
            estimate_burst_rate(bursts),
        "median_raw_burst_size":
            median_burst_size,
        "median_burst_gap_s":
            median_burst_gap,
        "video_fps_metadata":
            video.get("fps"),
        "video_frames_written":
            video.get("frames_written"),
        "video_start_unix":
            video.get("start_unix"),
        "video_stop_unix":
            video.get("stop_unix"),
        "anomalous_records":
            anomalies,
        "always_zero_positions_in_this_file":
            [
                int(i)
                for i in np.flatnonzero(
                    zero_counts
                    == total_iq_records
                )
            ]
            if total_iq_records
            else [],
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.RAW_CSI_DIR,
    )

    args = parser.parse_args()

    config.ensure_directories()

    if args.input_dir.resolve() == config.RAW_CSI_DIR.resolve():
        from .raw_paths import iter_raw_csi

        files = iter_raw_csi(args.input_dir)
    else:
        files = sorted(args.input_dir.rglob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No JSON files under {args.input_dir} "
            "(checked day1/day2/day3 subfolders too)"
        )

    results = [
        analyze_file(path)
        for path in files
    ]

    out = (
        config.QA_DIR
        / "csi_characteristics.json"
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nCSI DATA CHARACTERISTICS\n"
        "========================"
    )

    for r in results:
        print(
            f"{r['file']}: "
            f"records={r['records_total']} | "
            f"snapshots={r['complete_valid_snapshots']} | "
            f"bursts={r['usable_bursts']} | "
            f"snapshot_rate="
            f"{r['estimated_complete_snapshot_rate_hz']:.2f} Hz | "
            f"burst_rate="
            f"{r['estimated_burst_rate_hz']:.2f} Hz | "
            f"median_burst="
            f"{r['median_raw_burst_size']:.1f}"
        )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
