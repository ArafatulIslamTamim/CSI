import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from . import config


@dataclass
class CompleteSnapshot:
    host_ts: float
    complex_csi: np.ndarray   # [3,53]
    raw_indices: tuple
    rssi: np.ndarray          # [3]
    snr: np.ndarray           # [3]


@dataclass
class Burst:
    center_ts: float
    snapshot_ts: np.ndarray   # [5]
    complex_csi: np.ndarray   # [5,3,53]
    rssi: np.ndarray          # [5,3]
    snr: np.ndarray           # [5,3]


def parse_complex_string(value) -> complex:
    if isinstance(value, complex):
        return value

    if isinstance(value, (int, float)):
        return complex(value)

    return complex(
        str(value)
        .strip()
        .replace("+-", "-")
    )


def load_csi_json(path: Path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"{path}: expected a JSON object."
        )

    metadata = obj.get("metadata", {})
    records = obj.get("records")

    if not isinstance(records, list):
        raise ValueError(
            f"{path}: missing records list."
        )

    return metadata, records


def is_valid_record(record: dict) -> bool:
    try:
        return (
            int(record.get("tx_ant", -1)) == config.TX_ANT
            and int(record.get("rx_ant", -1)) in config.RX_INDICES
            and int(record.get("bw", -1)) == config.VALID_BW
            and int(record.get("samples", -1)) == config.VALID_SAMPLES
            and int(record.get("mode", -1)) == config.VALID_MODE
            and int(record.get("ext", -1)) == config.VALID_EXT
            and isinstance(record.get("iq"), list)
            and len(record["iq"]) == config.RAW_NUM_SUBCARRIERS
            and record.get("host_ts") is not None
        )
    except Exception:
        return False


def parse_record_iq(record: dict) -> np.ndarray:
    values = np.asarray(
        [
            parse_complex_string(v)
            for v in record["iq"]
        ],
        dtype=np.complex64,
    )

    if values.shape != (
        config.RAW_NUM_SUBCARRIERS,
    ):
        raise ValueError(
            f"Expected 64 IQ values, got {values.shape}."
        )

    # Remove permanent 0+0j region 27..37.
    return values[
        np.asarray(
            config.USABLE_SUBCARRIER_INDICES,
            dtype=np.int64,
        )
    ]


def group_complete_snapshots(
    records: List[dict],
) -> List[CompleteSnapshot]:
    """
    Group the valid record stream into complete:
        Rx0 + Rx1 + Rx2
    snapshots.

    The files normally follow:
        Rx0, Rx1, Rx2, Rx0, Rx1, Rx2, ...

    We tolerate discarded anomalous records by accumulating the next
    unique Rx0/Rx1/Rx2 set, but duplicate Rx before completion resets
    the partial group.
    """
    valid = [
        r
        for r in records
        if is_valid_record(r)
    ]

    valid.sort(
        key=lambda r: (
            int(r.get("index", 10**18)),
            float(r["host_ts"]),
        )
    )

    snapshots = []
    current: Dict[int, dict] = {}

    for record in valid:
        rx = int(record["rx_ant"])

        if rx in current:
            current = {}

        current[rx] = record

        if all(
            rx_id in current
            for rx_id in config.RX_INDICES
        ):
            ordered = [
                current[rx_id]
                for rx_id in config.RX_INDICES
            ]

            times = np.asarray(
                [
                    float(r["host_ts"])
                    for r in ordered
                ],
                dtype=np.float64,
            )

            complex_csi = np.stack(
                [
                    parse_record_iq(r)
                    for r in ordered
                ],
                axis=0,
            )

            snapshots.append(
                CompleteSnapshot(
                    host_ts=float(np.median(times)),
                    complex_csi=complex_csi,
                    raw_indices=tuple(
                        int(r.get("index", -1))
                        for r in ordered
                    ),
                    rssi=np.asarray(
                        [
                            float(r.get("rssi", np.nan))
                            for r in ordered
                        ],
                        dtype=np.float32,
                    ),
                    snr=np.asarray(
                        [
                            float(r.get("snr", np.nan))
                            for r in ordered
                        ],
                        dtype=np.float32,
                    ),
                )
            )

            current = {}

    return snapshots


def _pad_or_trim_burst(
    group: List[CompleteSnapshot],
) -> Burst:
    """
    Force one burst to exactly 5 snapshots.

    If >5:
        choose five snapshots closest to burst center.

    If 3 or 4:
        edge-pad by repeating nearest boundary snapshots.
    """
    times = np.asarray(
        [s.host_ts for s in group],
        dtype=np.float64,
    )

    center = float(np.median(times))

    if len(group) > config.SNAPSHOTS_PER_BURST:
        order = np.argsort(
            np.abs(times - center)
        )[:config.SNAPSHOTS_PER_BURST]

        order = np.sort(order)

        selected = [
            group[int(i)]
            for i in order
        ]

    else:
        selected = list(group)

    while len(selected) < config.SNAPSHOTS_PER_BURST:
        # Alternate padding around both sides.
        if len(selected) % 2 == 1:
            selected.append(selected[-1])
        else:
            selected.insert(0, selected[0])

    selected = selected[
        :config.SNAPSHOTS_PER_BURST
    ]

    return Burst(
        center_ts=float(
            np.median(
                [s.host_ts for s in selected]
            )
        ),
        snapshot_ts=np.asarray(
            [s.host_ts for s in selected],
            dtype=np.float64,
        ),
        complex_csi=np.stack(
            [s.complex_csi for s in selected],
            axis=0,
        ).astype(np.complex64),
        rssi=np.stack(
            [s.rssi for s in selected],
            axis=0,
        ).astype(np.float32),
        snr=np.stack(
            [s.snr for s in selected],
            axis=0,
        ).astype(np.float32),
    )


def group_bursts(
    snapshots: List[CompleteSnapshot],
) -> List[Burst]:
    if not snapshots:
        return []

    snapshots = sorted(
        snapshots,
        key=lambda s: s.host_ts,
    )

    groups = []
    current = [snapshots[0]]

    for snapshot in snapshots[1:]:
        gap = (
            snapshot.host_ts
            - current[-1].host_ts
        )

        if gap > config.BURST_GAP_THRESHOLD_S:
            groups.append(current)
            current = [snapshot]
        else:
            current.append(snapshot)

    groups.append(current)

    bursts = []

    for group in groups:
        if not (
            config.MIN_SNAPSHOTS_PER_BURST
            <= len(group)
            <= config.MAX_SNAPSHOTS_PER_BURST
        ):
            continue

        bursts.append(
            _pad_or_trim_burst(group)
        )

    return bursts


def amplitude_from_complex(
    complex_csi: np.ndarray,
) -> np.ndarray:
    return np.abs(
        complex_csi
    ).astype(np.float32)


def estimate_snapshot_rate(
    snapshots: List[CompleteSnapshot],
) -> float:
    if len(snapshots) < 3:
        return float("nan")

    ts = np.asarray(
        [s.host_ts for s in snapshots],
        dtype=np.float64,
    )

    duration = ts[-1] - ts[0]

    if duration <= 0:
        return float("nan")

    return float(
        (len(ts) - 1)
        / duration
    )


def estimate_burst_rate(
    bursts: List[Burst],
) -> float:
    if len(bursts) < 3:
        return float("nan")

    ts = np.asarray(
        [b.center_ts for b in bursts],
        dtype=np.float64,
    )

    duration = ts[-1] - ts[0]

    if duration <= 0:
        return float("nan")

    return float(
        (len(ts) - 1)
        / duration
    )


def parse_capture(path: Path):
    metadata, records = load_csi_json(path)

    snapshots = group_complete_snapshots(
        records
    )

    bursts = group_bursts(
        snapshots
    )

    return metadata, records, snapshots, bursts
