import torch

from src import config
from src.models import (
    CNNFCSAM,
    CNNGRUSAM,
    CNNGRUMultiTaskSAM,
    TransformerSAM,
    unpack_model_output,
)


def main():
    x = torch.randn(
        2,
        config.BURSTS_PER_WINDOW,
        config.SNAPSHOTS_PER_BURST,
        config.NUM_RX,
        config.NUM_USABLE_SUBCARRIERS,
    )

    expected_sam = (
        2,
        2,
        config.NUM_KEYPOINTS,
        config.NUM_KEYPOINTS,
    )

    for model in [CNNFCSAM(), CNNGRUSAM(), TransformerSAM()]:
        sam, logits = unpack_model_output(model(x))
        assert tuple(sam.shape) == expected_sam
        assert logits is None

    model = CNNGRUMultiTaskSAM()
    sam, logits = unpack_model_output(model(x))
    assert tuple(sam.shape) == expected_sam
    assert tuple(logits.shape) == (2, config.NUM_POSE_CLASSES)

    print("All model shape tests passed.")


if __name__ == "__main__":
    main()
