"""Session-balanced hierarchical sampling for training."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler

from . import config


def _session_day(session_id: str) -> str:
    return str(session_id).split("_", 1)[0]


def _session_distance(session_id: str) -> str:
    s = str(session_id).lower()
    if "far_front" in s:
        return "far"
    if "center_front" in s:
        return "center"
    return "unknown"


class SessionBalancedSampler(Sampler):
    """
    Hierarchical sampling: pose -> day -> distance -> subject -> session -> window.

    Batch composition defaults: 20% empty, 40% sit, 40% stand;
    occupied batches 50% center / 50% far.
    """

    def __init__(
        self,
        dataset,
        batch_size: int | None = None,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.batch_size = int(
            batch_size or config.BATCH_SIZE
        )
        self.rng = np.random.default_rng(
            seed if seed is not None else config.RANDOM_SEED
        )

        empty_frac = float(
            getattr(config, "BATCH_EMPTY_FRAC", 0.20)
        )
        self.n_empty = max(1, int(round(self.batch_size * empty_frac)))
        self.n_occupied = self.batch_size - self.n_empty
        self.n_sit = self.n_occupied // 2
        self.n_stand = self.n_occupied - self.n_sit

        self._tree = self._build_tree()

    def _build_tree(self):
        tree = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(list)
                    )
                )
            )
        )
        for local_i, global_idx in enumerate(self.dataset.indices):
            pose = str(self.dataset.pose_name[int(global_idx)]).lower()
            session = str(self.dataset.session_id[int(global_idx)])
            day = _session_day(session)
            subj = str(self.dataset.subject_id[int(global_idx)])
            dist = (
                "n/a"
                if pose == config.EMPTY_POSE_NAME
                else _session_distance(session)
            )
            tree[pose][day][dist][subj][session].append(int(local_i))
        return tree

    def _pick(self, pose: str, n: int) -> list[int]:
        picked = []
        pose_tree = self._tree.get(pose, {})
        if not pose_tree:
            return picked

        for _ in range(n):
            day = self.rng.choice(list(pose_tree.keys()))
            dist_tree = pose_tree[day]
            if pose != config.EMPTY_POSE_NAME:
                dists = [d for d in ("center", "far") if d in dist_tree]
                if not dists:
                    dists = list(dist_tree.keys())
                dist = self.rng.choice(dists)
            else:
                dist = self.rng.choice(list(dist_tree.keys()))
            subj_tree = dist_tree[dist]
            subj = self.rng.choice(list(subj_tree.keys()))
            sess_tree = subj_tree[subj]
            session = self.rng.choice(list(sess_tree.keys()))
            idx = int(
                self.rng.choice(sess_tree[session])
            )
            picked.append(idx)
        return picked

    def __iter__(self):
        n = len(self.dataset)
        n_batches = max(1, n // self.batch_size)
        for _ in range(n_batches):
            batch = []
            batch.extend(
                self._pick(config.EMPTY_POSE_NAME, self.n_empty)
            )
            batch.extend(
                self._pick("sitting", self.n_sit)
            )
            batch.extend(
                self._pick("standing", self.n_stand)
            )
            while len(batch) < self.batch_size:
                batch.append(
                    int(self.rng.integers(0, len(self.dataset)))
                )
            self.rng.shuffle(batch)
            yield from batch[: self.batch_size]

    def __len__(self):
        return (
            len(self.dataset)
            // self.batch_size
        ) * self.batch_size
