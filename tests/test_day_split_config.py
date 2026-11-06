from src import config


def main():
    assert config.SPLIT_MODE == "fixed_day"
    assert config.INCLUDE_EMPTY_ROOM_IN_DATASET is False
    assert config.POSE_CLASSES == ("sitting", "standing")
    assert config.EMPTY_CLASS_INDEX == -1
    assert config.MODEL_NAME in ("cnn_gru_multitask_sam", "tcn_multitask_sam")
    assert config.WINDOW_MODE in ("burst", "time_causal")
    assert config.FIXED_TRAIN_DAYS == ("day1", "day2", "day5")
    assert config.FIXED_VAL_DAYS == ("day3",)
    assert config.FIXED_TEST_DAYS == ("day4",)

    overlap = (
        set(config.FIXED_TRAIN_DAYS)
        & set(config.FIXED_VAL_DAYS)
        | set(config.FIXED_TRAIN_DAYS)
        & set(config.FIXED_TEST_DAYS)
        | set(config.FIXED_VAL_DAYS)
        & set(config.FIXED_TEST_DAYS)
    )
    assert not overlap

    print("Day split configuration test passed.")
    print("EXPERIMENT_TAG:", config.EXPERIMENT_TAG)
    print("POSE_CLASSES:", config.POSE_CLASSES)


if __name__ == "__main__":
    main()
