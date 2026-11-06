import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


def _get_local_norm_mode():
    """
    Local CSI normalization applied before the existing
    training-only Rx x subcarrier normalization.

    Supported modes:
      - "none"
      - "per_snapshot_subcarrier"
    """
    return str(
        getattr(
            config,
            "CSI_LOCAL_NORMALIZATION",
            "none",
        )
    ).lower()


def _apply_local_csi_normalization(
    csi,
):
    """
    Normalize each Rx independently across the 53 subcarriers.

    Works for:
      [N, bursts, snapshots, Rx, subcarriers]
    and:
      [bursts, snapshots, Rx, subcarriers]
    """
    mode = _get_local_norm_mode()

    if mode == "none":
        return csi.astype(
            np.float32,
            copy=False,
        )

    if mode == "per_snapshot_subcarrier":
        local_mean = csi.mean(
            axis=-1,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        local_std = csi.std(
            axis=-1,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        local_std = np.maximum(
            local_std,
            config.CSI_NORM_EPS,
        ).astype(np.float32)

        return (
            (csi - local_mean)
            / local_std
        ).astype(np.float32)

    raise ValueError(
        "Unknown CSI_LOCAL_NORMALIZATION="
        f"{mode!r}. Use 'none' or "
        "'per_snapshot_subcarrier'."
    )


def compute_and_save_csi_stats(
    train_indices,
):
    data = np.load(
        config.DATASET_FILE
    )

    idx = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    csi = data["csi"][
        idx
    ].astype(np.float32)

    # First remove per-snapshot/per-Rx amplitude offset and scale
    # when the experiment enables local normalization.
    csi = _apply_local_csi_normalization(
        csi
    )

    # Mean/std over all batch and time/burst dimensions; keep Rx x subcarrier.
    reduce_axes = tuple(range(csi.ndim - 2))
    mean = csi.mean(axis=reduce_axes, dtype=np.float64).astype(np.float32)
    std = csi.std(axis=reduce_axes, dtype=np.float64).astype(np.float32)

    std = np.maximum(
        std,
        config.CSI_NORM_EPS,
    ).astype(np.float32)

    np.savez_compressed(
        config.CSI_NORM_FILE,
        mean=mean,
        std=std,
        local_normalization=np.asarray(
            _get_local_norm_mode()
        ),
    )

    print(
        "[normalization] local mode="
        f"{_get_local_norm_mode()} | "
        "training-only mean/std shape: "
        f"{mean.shape}"
    )


def compute_adaptive_stats(
    indices,
):
    """
    AdaBN-style test-time adaptation (P3): recompute the global
    Rx x subcarrier normalization statistics from the (unlabeled) CSI of
    the given samples instead of the training split. Uses only CSI
    amplitudes, never labels.
    """
    data = np.load(config.DATASET_FILE)

    idx = np.asarray(indices, dtype=np.int64)
    csi = data["csi"][idx].astype(np.float32)
    csi = _apply_local_csi_normalization(csi)

    mean = csi.mean(
        axis=tuple(range(csi.ndim - 2)),
        dtype=np.float64,
    ).astype(np.float32)

    std = csi.std(
        axis=tuple(range(csi.ndim - 2)),
        dtype=np.float64,
    ).astype(np.float32)

    std = np.maximum(
        std,
        config.CSI_NORM_EPS,
    ).astype(np.float32)

    return mean, std


def _augment_csi(
    csi,
    rng,
):
    """
    Training-only CSI augmentation (P3), applied after normalization:
    per-Rx gain jitter, random subcarrier masking, additive Gaussian noise.

    csi: [bursts, snapshots, Rx, subcarriers]
    """
    rx_axis = csi.ndim - 2
    if config.AUG_RX_GAIN_STD > 0.0:
        gain_shape = [1] * rx_axis + [csi.shape[rx_axis], 1]
        gain = 1.0 + rng.normal(
            0.0,
            config.AUG_RX_GAIN_STD,
            size=tuple(gain_shape),
        ).astype(np.float32)
        csi = csi * gain

    if (
        config.AUG_SUBCARRIER_DROP_PROB > 0.0
        and rng.random() < config.AUG_SUBCARRIER_DROP_PROB
    ):
        num_subcarriers = csi.shape[-1]
        num_drop = max(
            1,
            int(round(num_subcarriers * config.AUG_SUBCARRIER_DROP_FRAC)),
        )
        drop = rng.choice(
            num_subcarriers,
            size=num_drop,
            replace=False,
        )
        csi[..., drop] = 0.0

    if config.AUG_NOISE_STD > 0.0:
        csi = csi + rng.normal(
            0.0,
            config.AUG_NOISE_STD,
            size=csi.shape,
        ).astype(np.float32)

    return csi.astype(np.float32, copy=False)


class CSIPoseDataset(
    Dataset
):
    def __init__(
        self,
        indices=None,
        augment=False,
    ):
        if not config.DATASET_FILE.exists():
            raise FileNotFoundError(
                config.DATASET_FILE
            )

        data = np.load(
            config.DATASET_FILE
        )

        self.csi = data[
            "csi"
        ].astype(np.float32)

        self.sam = data[
            "sam"
        ].astype(np.float32)

        self.keypoints = data[
            "keypoints"
        ].astype(np.float32)

        self.confidence = data[
            "confidence"
        ].astype(np.float32)

        self.frame_width = data[
            "frame_width"
        ].astype(np.float32)

        self.frame_height = data[
            "frame_height"
        ].astype(np.float32)

        self.session_id = data[
            "session_id"
        ].astype(str)

        self.subject_id = data[
            "subject_id"
        ].astype(str)

        self.pose_name = data[
            "pose_name"
        ].astype(str)

        if indices is None:
            self.indices = np.arange(
                len(self.csi),
                dtype=np.int64,
            )
        else:
            self.indices = np.asarray(
                indices,
                dtype=np.int64,
            )

        # Augmentation is opt-in and additionally gated by config so
        # evaluation loaders never receive augmented samples.
        self.augment = bool(augment) and bool(config.AUGMENT_CSI)
        self._augment_rng = np.random.default_rng(config.RANDOM_SEED)

        self._subject_to_class = {
            subject: i
            for i, subject in enumerate(config.SUBJECT_IDS)
        }

        self.mean = None
        self.std = None

        if config.NORMALIZE_CSI:
            if not config.CSI_NORM_FILE.exists():
                raise FileNotFoundError(
                    "Normalization statistics missing. "
                    "Run training."
                )

            stats = np.load(
                config.CSI_NORM_FILE
            )

            self.mean = stats[
                "mean"
            ].astype(np.float32)

            self.std = stats[
                "std"
            ].astype(np.float32)

            saved_mode = (
                str(
                    stats[
                        "local_normalization"
                    ].item()
                ).lower()
                if "local_normalization" in stats.files
                else "none"
            )

            current_mode = (
                _get_local_norm_mode()
            )

            if saved_mode != current_mode:
                raise RuntimeError(
                    "CSI normalization mode mismatch: "
                    f"stats were created with {saved_mode!r}, "
                    f"but config requests {current_mode!r}. "
                    "Run training again to recompute "
                    "training-only normalization statistics."
                )

    def __len__(self):
        return len(
            self.indices
        )

    def __getitem__(
        self,
        item,
    ):
        idx = int(
            self.indices[item]
        )

        csi = self.csi[
            idx
        ].copy()

        # Local normalization is deterministic and uses only
        # the current CSI sample, so it introduces no
        # train/validation/test leakage.
        csi = _apply_local_csi_normalization(
            csi
        )

        if config.NORMALIZE_CSI:
            if csi.ndim == 3:
                csi = (
                    csi - self.mean
                ) / (
                    self.std + config.CSI_NORM_EPS
                )
            else:
                csi = (
                    csi
                    - self.mean[
                        None,
                        None,
                        :,
                        :,
                    ]
                ) / (
                    self.std[
                        None,
                        None,
                        :,
                        :,
                    ]
                    + config.CSI_NORM_EPS
                )
            from .csi_features import clip_normalized
            csi = clip_normalized(csi)

        if self.augment:
            csi = _augment_csi(
                csi,
                self._augment_rng,
            )

        pose_name = str(self.pose_name[idx]).lower()
        is_empty = pose_name == config.EMPTY_POSE_NAME

        if is_empty:
            posture_class = -1
        elif pose_name in config.POSE_TO_CLASS:
            posture_class = config.POSE_TO_CLASS[pose_name]
        else:
            raise ValueError(
                f"Unknown pose_name={pose_name!r}. "
                f"Expected empty or {tuple(config.POSE_TO_CLASS)}"
            )

        return {
            "csi":
                torch.from_numpy(
                    csi.astype(np.float32)
                ),
            "sam":
                torch.from_numpy(
                    self.sam[idx]
                ),
            "keypoints":
                torch.from_numpy(
                    self.keypoints[idx]
                ),
            "confidence":
                torch.from_numpy(
                    self.confidence[idx]
                ),
            "pose_class":
                torch.tensor(
                    posture_class,
                    dtype=torch.long,
                ),
            "posture_class":
                torch.tensor(
                    max(posture_class, 0),
                    dtype=torch.long,
                ),
            "is_empty":
                torch.tensor(is_empty, dtype=torch.bool),
            "presence":
                torch.tensor(not is_empty, dtype=torch.float32),
            # -1 for empty-room captures (subject "000") and any subject
            # outside SUBJECT_IDS; the adversarial loss masks those out.
            "subject_class":
                torch.tensor(
                    self._subject_to_class.get(
                        str(self.subject_id[idx]),
                        -1,
                    ),
                    dtype=torch.long,
                ),
            "frame_width":
                torch.tensor(
                    self.frame_width[idx],
                    dtype=torch.float32,
                ),
            "frame_height":
                torch.tensor(
                    self.frame_height[idx],
                    dtype=torch.float32,
                ),
            "global_index":
                idx,
        }


def _split_units(
    units,
    rng,
):
    unique = np.asarray(
        sorted(
            np.unique(units)
        )
    )

    rng.shuffle(
        unique
    )

    if len(unique) < 3:
        raise ValueError(
            "Need at least 3 independent units "
            "for train/val/test."
        )

    n = len(unique)

    n_test = max(
        1,
        int(round(
            n * config.TEST_RATIO
        )),
    )

    n_val = max(
        1,
        int(round(
            n * config.VAL_RATIO
        )),
    )

    n_train = (
        n - n_val - n_test
    )

    if n_train < 1:
        raise ValueError(
            "Not enough independent units."
        )

    train = unique[
        :n_train
    ]

    val = unique[
        n_train:
        n_train+n_val
    ]

    test = unique[
        n_train+n_val:
    ]

    return train, val, test


def _session_day(session_id: str) -> str:
    """Leading day folder token from a session id (e.g. day4_005_standing_...)."""
    return str(session_id).split("_", 1)[0]


def create_split_indices(
    force=False,
):
    if (
        config.SPLIT_FILE.exists()
        and not force
    ):
        obj = np.load(
            config.SPLIT_FILE
        )

        return (
            obj["train"],
            obj["val"],
            obj["test"],
        )

    data = np.load(
        config.DATASET_FILE
    )

    subjects = data[
        "subject_id"
    ].astype(str)

    sessions = data[
        "session_id"
    ].astype(str)

    poses = data[
        "pose_name"
    ].astype(str)

    rng = np.random.default_rng(
        config.RANDOM_SEED
    )

    include_empty = getattr(
        config,
        "INCLUDE_EMPTY_ROOM_IN_DATASET",
        True,
    )

    # Empty-room samples are assigned at session level (config lists),
    # never through the subject-unit logic: subject "000" would otherwise
    # fall entirely into a single split.
    is_empty = (
        poses == config.EMPTY_POSE_NAME
    )

    if not include_empty:
        is_empty = np.zeros(
            len(poses),
            dtype=bool,
        )

    empty_split_indices = {
        "train": [],
        "val": [],
        "test": [],
    }

    if include_empty and (
        poses == config.EMPTY_POSE_NAME
    ).any():
        empty_assignment = {}

        for name, session_list in [
            ("train", config.EMPTY_TRAIN_SESSIONS),
            ("val", config.EMPTY_VAL_SESSIONS),
            ("test", config.EMPTY_TEST_SESSIONS),
        ]:
            for session in session_list:
                empty_assignment[str(session)] = name

        empty_sessions_present = set(
            np.unique(sessions[is_empty]).tolist()
        )

        day_split = getattr(config, "EMPTY_DAY_SPLIT", {})

        for session in sorted(empty_sessions_present):
            if session in empty_assignment:
                continue
            day = str(session).split("_", 1)[0]
            if day in day_split:
                empty_assignment[session] = day_split[day]
                continue
            raise ValueError(
                "Empty-room session without a split assignment "
                f"in config: {session}"
            )

        for idx in np.flatnonzero(is_empty):
            split_name = empty_assignment[
                str(sessions[idx])
            ]
            empty_split_indices[split_name].append(int(idx))

    if config.SPLIT_MODE == "fixed_subject":
        units = subjects

        unique_subjects = set(
            np.unique(subjects).astype(str).tolist()
        ) - {config.EMPTY_SUBJECT_ID}

        test_units = np.asarray(
            config.FIXED_TEST_SUBJECTS,
            dtype=str,
        )

        val_units = np.asarray(
            config.FIXED_VAL_SUBJECTS,
            dtype=str,
        )

        requested = set(
            test_units.tolist()
            + val_units.tolist()
        )

        missing = requested - unique_subjects
        if missing:
            raise ValueError(
                f"Fixed split subjects not found in dataset: {sorted(missing)}"
            )

        overlap = set(test_units.tolist()) & set(val_units.tolist())
        if overlap:
            raise ValueError(
                f"Subjects cannot be both validation and test: {sorted(overlap)}"
            )

        train_units = np.asarray(
            sorted(
                unique_subjects
                - set(test_units.tolist())
                - set(val_units.tolist())
            ),
            dtype=str,
        )

        if len(train_units) == 0:
            raise ValueError(
                "Fixed split leaves no training subjects."
            )

    elif config.SPLIT_MODE == "subject":
        units = subjects

        (
            train_units,
            val_units,
            test_units,
        ) = _split_units(
            units[~is_empty],
            rng,
        )

    elif config.SPLIT_MODE == "session":
        units = sessions

        (
            train_units,
            val_units,
            test_units,
        ) = _split_units(
            units[~is_empty],
            rng,
        )

    elif config.SPLIT_MODE == "fixed_day":
        days = np.asarray(
            [_session_day(s) for s in sessions],
            dtype=str,
        )

        train_units = np.asarray(
            config.FIXED_TRAIN_DAYS,
            dtype=str,
        )
        val_units = np.asarray(
            config.FIXED_VAL_DAYS,
            dtype=str,
        )
        test_units = np.asarray(
            config.FIXED_TEST_DAYS,
            dtype=str,
        )

        requested = (
            set(train_units.tolist())
            | set(val_units.tolist())
            | set(test_units.tolist())
        )
        present = set(np.unique(days).tolist())
        missing = requested - present
        if missing:
            raise ValueError(
                f"Fixed day split: days not in dataset: {sorted(missing)}"
            )

        overlap = (
            set(train_units.tolist())
            & set(val_units.tolist())
            | set(train_units.tolist())
            & set(test_units.tolist())
            | set(val_units.tolist())
            & set(test_units.tolist())
        )
        if overlap:
            raise ValueError(
                f"Days cannot appear in more than one split: {sorted(overlap)}"
            )

        eligible = ~is_empty
        train_idx = np.flatnonzero(
            eligible & np.isin(days, train_units)
        )
        val_idx = np.flatnonzero(
            eligible & np.isin(days, val_units)
        )
        test_idx = np.flatnonzero(
            eligible & np.isin(days, test_units)
        )

        unassigned = eligible & ~np.isin(
            days,
            np.concatenate([train_units, val_units, test_units]),
        )
        if unassigned.any():
            extra = sorted(
                np.unique(days[unassigned]).tolist()
            )
            raise ValueError(
                "Samples on days without a split assignment: "
                f"{extra}"
            )

        if include_empty:
            train_idx = np.sort(
                np.concatenate([
                    train_idx,
                    np.asarray(
                        empty_split_indices["train"],
                        dtype=np.int64,
                    ),
                ])
            )
            val_idx = np.sort(
                np.concatenate([
                    val_idx,
                    np.asarray(
                        empty_split_indices["val"],
                        dtype=np.int64,
                    ),
                ])
            )
            test_idx = np.sort(
                np.concatenate([
                    test_idx,
                    np.asarray(
                        empty_split_indices["test"],
                        dtype=np.int64,
                    ),
                ])
            )

        print(
            f"[split] mode={config.SPLIT_MODE}"
        )
        print(
            f"[split] train days={train_units.tolist()}"
        )
        print(
            f"[split] val days={val_units.tolist()}"
        )
        print(
            f"[split] test days={test_units.tolist()}"
        )
        for name, idx in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            day_counts = {}
            for d in days[idx]:
                day_counts[str(d)] = day_counts.get(str(d), 0) + 1
            print(
                f"[split] {name}: "
                f"{len(idx)} samples | "
                f"poses={sorted(np.unique(poses[idx]).tolist())} | "
                f"days={day_counts}"
            )

        np.savez(
            config.SPLIT_FILE,
            train=train_idx.astype(np.int64),
            val=val_idx.astype(np.int64),
            test=test_idx.astype(np.int64),
        )

        return (
            train_idx,
            val_idx,
            test_idx,
        )

    elif config.SPLIT_MODE == "loso_day":
        holdout = str(getattr(config, "LODO_HOLDOUT_DAY", "day1"))
        days = np.asarray(
            [_session_day(s) for s in sessions],
            dtype=str,
        )
        train_idx = np.flatnonzero(days != holdout)
        val_idx = np.flatnonzero(days == holdout)
        test_idx = np.asarray([], dtype=np.int64)

        print(f"[split] mode=loso_day holdout={holdout}")
        print(f"[split] train: {len(train_idx)} val: {len(val_idx)}")

        np.savez(
            config.SPLIT_FILE,
            train=train_idx.astype(np.int64),
            val=val_idx.astype(np.int64),
            test=test_idx.astype(np.int64),
        )
        return train_idx, val_idx, test_idx

    elif config.SPLIT_MODE == "final_all_days":
        all_idx = np.arange(len(sessions), dtype=np.int64)
        train_idx = all_idx
        val_idx = np.asarray([], dtype=np.int64)
        test_idx = np.asarray([], dtype=np.int64)
        print("[split] mode=final_all_days (all samples in train)")
        np.savez(
            config.SPLIT_FILE,
            train=train_idx,
            val=val_idx,
            test=test_idx,
        )
        return train_idx, val_idx, test_idx

    else:
        raise ValueError(
            "Use SPLIT_MODE='fixed_subject', 'fixed_day', 'loso_day', "
            "'final_all_days', 'subject', or 'session'."
        )

    train_idx = np.flatnonzero(
        np.isin(
            units,
            train_units,
        )
    )

    val_idx = np.flatnonzero(
        np.isin(
            units,
            val_units,
        )
    )

    test_idx = np.flatnonzero(
        np.isin(
            units,
            test_units,
        )
    )

    # Append the session-assigned empty-room samples.
    if include_empty:
        train_idx = np.sort(
            np.concatenate([
                train_idx,
                np.asarray(
                    empty_split_indices["train"],
                    dtype=np.int64,
                ),
            ])
        )

        val_idx = np.sort(
            np.concatenate([
                val_idx,
                np.asarray(
                    empty_split_indices["val"],
                    dtype=np.int64,
                ),
            ])
        )

        test_idx = np.sort(
            np.concatenate([
                test_idx,
                np.asarray(
                    empty_split_indices["test"],
                    dtype=np.int64,
                ),
            ])
        )

    print(
        f"[split] mode={config.SPLIT_MODE}"
    )

    print(
        f"[split] train units={train_units.tolist()}"
    )

    print(
        f"[split] val units={val_units.tolist()}"
    )

    print(
        f"[split] test units={test_units.tolist()}"
    )

    for name, idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        print(
            f"[split] {name}: "
            f"{len(idx)} samples | "
            f"poses="
            f"{sorted(np.unique(poses[idx]).tolist())}"
        )

    np.savez(
        config.SPLIT_FILE,
        train=train_idx.astype(np.int64),
        val=val_idx.astype(np.int64),
        test=test_idx.astype(np.int64),
    )

    return (
        train_idx,
        val_idx,
        test_idx,
    )


def load_split_indices():
    if not config.SPLIT_FILE.exists():
        return create_split_indices()

    obj = np.load(
        config.SPLIT_FILE
    )

    return (
        obj["train"],
        obj["val"],
        obj["test"],
    )
