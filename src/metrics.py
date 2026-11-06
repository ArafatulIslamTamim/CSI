import numpy as np


def bbox_diagonal(
    xy,
):
    x = xy[:, 0]
    y = xy[:, 1]

    return float(
        np.hypot(
            x.max() - x.min(),
            y.max() - y.min(),
        )
    )


def normalized_pck(
    predicted_xy,
    ground_truth_xy,
    alpha,
):
    pred = np.asarray(
        predicted_xy,
        dtype=np.float32,
    )

    gt = np.asarray(
        ground_truth_xy,
        dtype=np.float32,
    )

    distances = np.linalg.norm(
        pred - gt,
        axis=-1,
    )

    thresholds = np.asarray(
        [
            alpha
            * max(
                bbox_diagonal(g),
                1e-8,
            )
            for g in gt
        ],
        dtype=np.float32,
    )[:, None]

    correct = (
        distances
        <= thresholds
    )

    overall = float(
        correct.mean()
    )

    per_joint = correct.mean(
        axis=0
    ).astype(np.float32)

    return overall, per_joint


def mpjpe(
    predicted_xy,
    ground_truth_xy,
):
    """
    Mean per-joint position error in normalized [0,1] coordinates.

    Returns (overall, per_joint[K]).
    """
    pred = np.asarray(predicted_xy, dtype=np.float32)
    gt = np.asarray(ground_truth_xy, dtype=np.float32)

    distances = np.linalg.norm(pred - gt, axis=-1)

    overall = float(distances.mean())
    per_joint = distances.mean(axis=0).astype(np.float32)

    return overall, per_joint


def pixel_mpjpe(
    predicted_xy,
    ground_truth_xy,
    widths,
    heights,
):
    """Mean per-joint position error in pixels."""
    pred = np.asarray(predicted_xy, dtype=np.float32).copy()
    gt = np.asarray(ground_truth_xy, dtype=np.float32).copy()

    widths = np.asarray(widths, dtype=np.float32)[:, None]
    heights = np.asarray(heights, dtype=np.float32)[:, None]

    pred[..., 0] *= widths
    pred[..., 1] *= heights
    gt[..., 0] *= widths
    gt[..., 1] *= heights

    distances = np.linalg.norm(pred - gt, axis=-1)

    return float(distances.mean())


def pixel_pck(
    predicted_xy,
    ground_truth_xy,
    widths,
    heights,
    thresholds_px,
):
    pred = np.asarray(
        predicted_xy,
        dtype=np.float32,
    )

    gt = np.asarray(
        ground_truth_xy,
        dtype=np.float32,
    )

    widths = np.asarray(
        widths,
        dtype=np.float32,
    )[:, None]

    heights = np.asarray(
        heights,
        dtype=np.float32,
    )[:, None]

    pred_px = pred.copy()
    gt_px = gt.copy()

    pred_px[..., 0] *= widths
    pred_px[..., 1] *= heights

    gt_px[..., 0] *= widths
    gt_px[..., 1] *= heights

    distance = np.linalg.norm(
        pred_px - gt_px,
        axis=-1,
    )

    result = {}

    for threshold in thresholds_px:
        correct = (
            distance
            <= float(threshold)
        )

        result[
            f"PCK@{int(threshold)}"
        ] = float(
            correct.mean()
        )

    return result
