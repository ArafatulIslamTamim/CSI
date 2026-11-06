import numpy as np


def generate_sam(keypoints_xy: np.ndarray) -> np.ndarray:
    """
    keypoints_xy:
        [K, 2] normalized x,y.

    Returns:
        [2, K, K]

    Diagonal = absolute coordinates.
    Off-diagonal = pairwise coordinate differences.
    """
    xy = np.asarray(keypoints_xy, dtype=np.float32)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"Expected keypoints [K,2], got {xy.shape}"
        )

    x = xy[:, 0]
    y = xy[:, 1]

    gx = x[:, None] - x[None, :]
    gy = y[:, None] - y[None, :]

    idx = np.arange(len(x))
    gx[idx, idx] = x
    gy[idx, idx] = y

    return np.stack(
        [gx, gy],
        axis=0,
    ).astype(np.float32)


def sam_to_keypoints(sam: np.ndarray) -> np.ndarray:
    """
    [2,K,K] -> [K,2]
    [B,2,K,K] -> [B,K,2]
    """
    arr = np.asarray(sam)

    if arr.ndim == 3:
        x = np.diag(arr[0])
        y = np.diag(arr[1])
        return np.stack([x, y], axis=-1)

    if arr.ndim == 4:
        x = np.diagonal(
            arr[:, 0],
            axis1=1,
            axis2=2,
        )

        y = np.diagonal(
            arr[:, 1],
            axis1=1,
            axis2=2,
        )

        return np.stack([x, y], axis=-1)

    raise ValueError(
        f"Unsupported SAM shape: {arr.shape}"
    )
