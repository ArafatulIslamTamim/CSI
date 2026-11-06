import argparse

import cv2
import numpy as np

from . import config


# COCO-17 edges
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


def draw_pose(
    image,
    xy,
    color,
):
    h, w = image.shape[:2]

    points = []

    for x, y in xy:
        px = int(
            round(
                np.clip(x, 0, 1)
                * w
            )
        )

        py = int(
            round(
                np.clip(y, 0, 1)
                * h
            )
        )

        points.append(
            (px, py)
        )

    for a, b in SKELETON:
        cv2.line(
            image,
            points[a],
            points[b],
            color,
            2,
            cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(
            image,
            point,
            4,
            color,
            -1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="sitting / standing / text in session name",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--prediction-only",
        action="store_true",
    )

    parser.add_argument(
        "--background",
        choices=[
            "white",
            "original",
        ],
        default="white",
    )

    args = parser.parse_args()

    if not config.TEST_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Run evaluation first."
        )

    data = np.load(
        config.DATASET_FILE
    )

    pred = np.load(
        config.TEST_PREDICTIONS_FILE
    )

    indices = pred[
        "global_index"
    ]

    predicted = pred[
        "predicted_keypoints"
    ]

    ground_truth = pred[
        "ground_truth_keypoints"
    ]

    sessions = data[
        "session_id"
    ].astype(str)

    frame_paths = data[
        "frame_path"
    ].astype(str)

    widths = data[
        "frame_width"
    ].astype(int)

    heights = data[
        "frame_height"
    ].astype(int)

    chosen = []

    for pred_idx, global_idx in enumerate(
        indices
    ):
        global_idx = int(
            global_idx
        )

        session = sessions[
            global_idx
        ]

        # Empty-room samples have no skeleton to draw.
        if "empty_room" in session.lower():
            continue

        if (
            args.pose
            and args.pose.lower()
            not in session.lower()
        ):
            continue

        chosen.append(
            (
                pred_idx,
                global_idx,
                session,
            )
        )

    chosen = chosen[
        :args.num_samples
    ]

    if not chosen:
        print(
            "No matching test samples."
        )
        return

    out_dir = (
        config.VISUALIZATION_DIR
        / config.EXPERIMENT_TAG
        / config.MODEL_NAME
        / (
            args.pose.lower()
            if args.pose
            else "all"
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        pred_idx,
        global_idx,
        session,
    ) in chosen:
        width = int(
            widths[global_idx]
        )

        height = int(
            heights[global_idx]
        )

        if args.background == "original":
            image = cv2.imread(
                frame_paths[
                    global_idx
                ]
            )

            if image is None:
                raise RuntimeError(
                    "Could not read original frame."
                )

        else:
            image = np.ones(
                (
                    height,
                    width,
                    3,
                ),
                dtype=np.uint8,
            ) * 255

        if not args.prediction_only:
            draw_pose(
                image,
                ground_truth[
                    pred_idx
                ],
                (0, 180, 0),
            )

        draw_pose(
            image,
            predicted[
                pred_idx
            ],
            (0, 0, 220),
        )

        text = (
            f"{session} | Pred=red"
        )

        if not args.prediction_only:
            text += " | GT=green"

        cv2.putText(
            image,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        out_path = (
            out_dir
            / (
                f"{session}_sample_"
                f"{global_idx:06d}.png"
            )
        )

        cv2.imwrite(
            str(out_path),
            image,
        )

        print(
            f"[viz] {out_path}"
        )


if __name__ == "__main__":
    main()
