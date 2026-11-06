import numpy as np

from . import config


def main():
    data = np.load(
        config.DATASET_FILE
    )

    print(
        "DATASET INSPECTION"
        "\n=================="
    )

    for key in data.files:
        value = data[key]

        print(
            f"{key:22s} "
            f"shape={value.shape} "
            f"dtype={value.dtype}"
        )

    print(
        "\nsubjects:",
        sorted(
            np.unique(
                data["subject_id"]
            ).astype(str).tolist()
        ),
    )

    print(
        "poses:",
        sorted(
            np.unique(
                data["pose_name"]
            ).astype(str).tolist()
        ),
    )

    print(
        "median sync error ms:",
        float(
            np.median(
                data["sync_error_s"]
            )
            * 1000.0
        ),
    )


if __name__ == "__main__":
    main()
