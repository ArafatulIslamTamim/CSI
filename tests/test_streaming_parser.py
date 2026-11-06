"""
The streaming (live) parser must produce exactly the same snapshots,
bursts, and windows as the offline parser the model was trained with.

Runs on every real capture JSON found in data/raw/csi (records are fed
in file order, like a live stream would arrive), plus a synthetic
stream so the test still checks the logic when no data is present.
"""

import numpy as np

from src import config
from src.csi_parser import (
    group_bursts,
    group_complete_snapshots,
    load_csi_json,
)
from src.live.pipeline import (
    SlidingWindowBuilder,
    StreamingBurstAssembler,
    StreamingSnapshotAssembler,
)
# Run as: python tests/test_streaming_parser.py (tests dir is on sys.path).
from test_parser_synthetic import record as synthetic_record


def stream_parse(records):
    """Feed records one by one through the streaming assemblers."""
    snapshot_assembler = StreamingSnapshotAssembler()
    burst_assembler = StreamingBurstAssembler()

    snapshots = []
    bursts = []

    for raw in records:
        snapshot = snapshot_assembler.push(raw)
        if snapshot is None:
            continue

        snapshots.append(snapshot)

        burst = burst_assembler.push(snapshot)
        if burst is not None:
            bursts.append(burst)

    final = burst_assembler.flush()
    if final is not None:
        bursts.append(final)

    return snapshots, bursts


def offline_windows(bursts):
    """Window logic as used by src.build_dataset.process_session."""
    windows = []

    for burst_idx in range(len(bursts)):
        start = burst_idx - config.BURSTS_PER_WINDOW + 1
        if start < 0:
            continue

        window = bursts[start:burst_idx + 1]
        span = window[-1].center_ts - window[0].center_ts
        if span > config.MAX_BURST_WINDOW_SPAN_S:
            continue

        windows.append(
            np.stack([b.complex_csi for b in window], axis=0)
        )

    return windows


def compare_capture(records, name):
    offline_snapshots = group_complete_snapshots(records)
    offline_bursts = group_bursts(offline_snapshots)

    stream_snapshots, stream_bursts = stream_parse(records)

    assert len(stream_snapshots) == len(offline_snapshots), (
        f"{name}: snapshot count {len(stream_snapshots)} "
        f"!= offline {len(offline_snapshots)}"
    )

    for a, b in zip(stream_snapshots, offline_snapshots):
        assert a.host_ts == b.host_ts
        assert np.array_equal(a.complex_csi, b.complex_csi)
        assert a.raw_indices == b.raw_indices

    assert len(stream_bursts) == len(offline_bursts), (
        f"{name}: burst count {len(stream_bursts)} "
        f"!= offline {len(offline_bursts)}"
    )

    for a, b in zip(stream_bursts, offline_bursts):
        assert a.center_ts == b.center_ts
        assert np.array_equal(a.snapshot_ts, b.snapshot_ts)
        assert np.array_equal(a.complex_csi, b.complex_csi)

    # Window equivalence: the sliding builder must reproduce the
    # dataset-build window slicing.
    expected_windows = offline_windows(offline_bursts)

    builder = SlidingWindowBuilder()
    stream_windows = []
    for burst in stream_bursts:
        window = builder.push(burst)
        if window is not None:
            stream_windows.append(window)

    assert len(stream_windows) == len(expected_windows), (
        f"{name}: window count {len(stream_windows)} "
        f"!= offline {len(expected_windows)}"
    )

    for a, b in zip(stream_windows, expected_windows):
        assert np.array_equal(a, b)

    return len(offline_bursts), len(expected_windows)


def synthetic_records():
    records = []
    index = 0

    for burst_base in np.arange(100.0, 102.0, 0.11):
        for snapshot in range(5):
            t = float(burst_base) + snapshot * 0.002
            for rx in (0, 1, 2):
                records.append(synthetic_record(index, rx, t))
                index += 1

    return records


def main():
    n_bursts, n_windows = compare_capture(
        synthetic_records(),
        "synthetic",
    )
    print(f"synthetic: {n_bursts} bursts, {n_windows} windows OK")

    capture_files = sorted(config.RAW_CSI_DIR.glob("*.json"))

    if not capture_files:
        print("No real captures found; synthetic check only.")
    else:
        for path in capture_files:
            _, records = load_csi_json(path)
            n_bursts, n_windows = compare_capture(records, path.stem)
            print(
                f"{path.stem}: {n_bursts} bursts, "
                f"{n_windows} windows OK"
            )

    print("Streaming parser equivalence test passed.")


if __name__ == "__main__":
    main()
