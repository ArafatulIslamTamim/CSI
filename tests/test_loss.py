import torch

from src import config
from src.losses import ConfidenceWeightedMultiTaskLoss


def main():
    pred = torch.zeros(
        2,
        2,
        config.NUM_KEYPOINTS,
        config.NUM_KEYPOINTS,
    )
    target = torch.ones_like(pred)
    xy = torch.ones(2, config.NUM_KEYPOINTS, 2)
    confidence = torch.ones(2, config.NUM_KEYPOINTS)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    pose_class = torch.tensor([0, 1], dtype=torch.long)

    loss = ConfidenceWeightedMultiTaskLoss()(
        pred,
        target,
        xy,
        confidence,
        pose_logits=logits,
        pose_class=pose_class,
    )

    assert float(loss["total"]) > 0
    assert float(loss["classification"]) > 0
    print("Multi-task loss test passed.")


if __name__ == "__main__":
    main()
