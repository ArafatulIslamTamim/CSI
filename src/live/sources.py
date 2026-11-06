"""
CSI record sources for live inference.

Every source is an iterable of raw record dicts in the same format the
offline parser consumes (bpi_csi_print_v1 "records" entries). The rest of
the live pipeline only depends on this interface, so a real transport
(SSH pipe / TCP forwarder from the BPI) can replace ReplaySource without
touching anything downstream.
"""

import time
from pathlib import Path

from ..csi_parser import load_csi_json


class CSISource:
    """Iterable of raw CSI record dicts."""

    #: capture metadata when the transport provides it (may stay empty).
    metadata: dict = {}

    def __iter__(self):
        raise NotImplementedError

    def close(self):
        pass


class ReplaySource(CSISource):
    """
    Stream a recorded capture JSON as if it were arriving live.

    Records are yielded in file order and paced by their host_ts deltas,
    so bursts and idle gaps reproduce the original timing. speed > 1
    replays faster; speed=0 disables pacing entirely (as fast as possible,
    useful for tests).
    """

    # Recording gaps (e.g. between captures) should not stall the replay
    # for longer than this.
    MAX_SLEEP_S = 1.0

    def __init__(self, json_path, speed=1.0):
        self.path = Path(json_path)
        self.speed = float(speed)

        self.metadata, self._records = load_csi_json(self.path)

    def __iter__(self):
        previous_ts = None

        for record in self._records:
            host_ts = record.get("host_ts")

            if (
                self.speed > 0
                and previous_ts is not None
                and host_ts is not None
            ):
                delta = (float(host_ts) - previous_ts) / self.speed

                if delta > 0:
                    time.sleep(min(delta, self.MAX_SLEEP_S))

            if host_ts is not None:
                previous_ts = float(host_ts)

            yield record
