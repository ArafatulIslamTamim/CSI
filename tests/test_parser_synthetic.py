import numpy as np

from src import config
from src.csi_parser import (
    parse_complex_string,
    group_complete_snapshots,
    group_bursts,
)


def record(
    index,
    rx,
    host_ts,
):
    iq = [
        "1+2j"
        for _ in range(64)
    ]

    for i in config.ZERO_SUBCARRIER_INDICES:
        iq[i] = "0+0j"

    return {
        "index": index,
        "host_ts": host_ts,
        "tx_ant": 0,
        "rx_ant": rx,
        "bw": 0,
        "samples": 64,
        "mode": 1,
        "ext": 0,
        "rssi": -55,
        "snr": 40,
        "iq": iq,
    }


def main():
    assert (
        parse_complex_string(
            "744+-213j"
        )
        == complex(
            744,
            -213,
        )
    )

    records = []

    index = 0

    for burst_base in [
        100.0,
        100.1,
    ]:
        for snapshot in range(5):
            t = (
                burst_base
                + snapshot
                * 0.00001
            )

            for rx in [0, 1, 2]:
                records.append(
                    record(
                        index,
                        rx,
                        t,
                    )
                )

                index += 1

    snapshots = group_complete_snapshots(
        records
    )

    assert len(snapshots) == 10

    assert snapshots[0].complex_csi.shape == (
        3,
        53,
    )

    bursts = group_bursts(
        snapshots
    )

    assert len(bursts) == 2

    assert bursts[0].complex_csi.shape == (
        5,
        3,
        53,
    )

    print(
        "Synthetic CSI parser test passed."
    )


if __name__ == "__main__":
    main()
