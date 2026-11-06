import numpy as np

from src.sam import (
    generate_sam,
    sam_to_keypoints,
)


def main():
    xy = np.array(
        [
            [0.2, 0.3],
            [0.6, 0.7],
            [0.4, 0.5],
        ],
        dtype=np.float32,
    )

    sam = generate_sam(
        xy
    )

    assert sam.shape == (
        2,
        3,
        3,
    )

    assert np.allclose(
        sam_to_keypoints(
            sam
        ),
        xy,
    )

    assert np.isclose(
        sam[
            0,
            0,
            1,
        ],
        -0.4,
    )

    print(
        "SAM test passed."
    )


if __name__ == "__main__":
    main()
