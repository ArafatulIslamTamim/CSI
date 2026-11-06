from src import config


def main():
    if config.SPLIT_MODE != "fixed_subject":
        print(
            "Skipping fixed_subject config test "
            f"(SPLIT_MODE={config.SPLIT_MODE!r})."
        )
        return

    assert config.SPLIT_MODE == "fixed_subject"

    # Validation/test subjects must exist and not overlap.
    assert len(config.FIXED_TEST_SUBJECTS) >= 1
    assert len(config.FIXED_VAL_SUBJECTS) >= 1
    assert not (
        set(config.FIXED_TEST_SUBJECTS)
        & set(config.FIXED_VAL_SUBJECTS)
    )

    # The empty-room subject must never appear in the subject split lists;
    # its sessions are assigned separately at session level.
    assert config.EMPTY_SUBJECT_ID not in config.FIXED_TEST_SUBJECTS
    assert config.EMPTY_SUBJECT_ID not in config.FIXED_VAL_SUBJECTS

    # Empty-room sessions are assigned by explicit EMPTY_*_SESSIONS lists,
    # or by EMPTY_DAY_SPLIT (day1→train, …) when those lists are empty.
    empty_lists = (
        config.EMPTY_TRAIN_SESSIONS,
        config.EMPTY_VAL_SESSIONS,
        config.EMPTY_TEST_SESSIONS,
    )
    all_sessions = [s for group in empty_lists for s in group]
    assert len(all_sessions) == len(set(all_sessions))
    assert hasattr(config, "EMPTY_DAY_SPLIT")
    if not all_sessions:
        assert set(config.EMPTY_DAY_SPLIT) >= {"day1", "day2", "day3"}

    # Person-ID split across days (not day-as-subject).
    assert config.FIXED_TEST_SUBJECTS
    assert config.FIXED_VAL_SUBJECTS
    for sid in list(config.FIXED_TEST_SUBJECTS) + list(config.FIXED_VAL_SUBJECTS):
        assert not str(sid).startswith("day"), (
            "FIXED_*_SUBJECTS must be person IDs (001..007), not day names"
        )

    # Class layout: sitting/standing keep their original indices and
    # "empty" is appended last.
    assert config.POSE_CLASSES[:2] == ("sitting", "standing")
    assert config.POSE_CLASSES[-1] == config.EMPTY_POSE_NAME
    assert config.EMPTY_CLASS_INDEX == len(config.POSE_CLASSES) - 1

    print("Fixed split configuration test passed.")
    print("Video FPS:", config.TARGET_VIDEO_FPS)
    print("CSI snapshots per frame:", config.SNAPSHOTS_PER_BURST)
    print("Test subject:", config.FIXED_TEST_SUBJECTS)
    print("Validation subject:", config.FIXED_VAL_SUBJECTS)
    print("Empty day split:", config.EMPTY_DAY_SPLIT)
    print("Empty sessions (train/val/test):", empty_lists)
    print("CLASS_LOSS_WEIGHTS:", getattr(config, "CLASS_LOSS_WEIGHTS", None))
    print("LAMBDA_CLASS:", config.LAMBDA_CLASS)
    print("EXPERIMENT_TAG:", config.EXPERIMENT_TAG)


if __name__ == "__main__":
    main()
