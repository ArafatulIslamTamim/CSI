"""
Tests for the experiment-plan features:
P1 root-relative representation, P2 geometry losses, P4 adversarial head,
P5 GCN decoder.
"""

import math

import numpy as np
import torch

from src import config
from src.geometry import (
    bone_lengths,
    build_sam_from_keypoints,
    joint_cosines,
    skeleton_adjacency,
)
from src.losses import (
    ConfidenceWeightedMultiTaskLoss,
    pose_root_and_scale,
    reconstruct_absolute_keypoints,
    sam_diagonal,
)
from src.models import build_model
from src.sam import generate_sam


def test_geometry_functions():
    k = config.NUM_KEYPOINTS
    xy = torch.zeros(1, k, 2)

    # Left leg: hip (11) at origin, knee (13) below, ankle (15) below that
    # -> straight leg, cos(angle at knee) = -1.
    xy[0, 11] = torch.tensor([0.5, 0.4])
    xy[0, 13] = torch.tensor([0.5, 0.6])
    xy[0, 15] = torch.tensor([0.5, 0.8])

    cosines = joint_cosines(xy)
    knee_index = 2  # (11, 13, 15) in ANGLE_TRIPLETS
    assert abs(cosines[0, knee_index].item() - (-1.0)) < 1e-5

    # Bend the knee to 90 degrees -> cos = 0.
    xy[0, 15] = torch.tensor([0.7, 0.6])
    cosines = joint_cosines(xy)
    assert abs(cosines[0, knee_index].item()) < 1e-5

    lengths = bone_lengths(xy)
    # Bone (11,13) has length 0.2.
    bone_index = 12  # (11,13) position in BONES
    assert abs(lengths[0, bone_index].item() - 0.2) < 1e-5

    adjacency = skeleton_adjacency(k)
    assert adjacency.shape == (k, k)
    assert torch.allclose(adjacency, adjacency.T)


def test_build_sam_matches_numpy():
    rng = np.random.default_rng(0)
    xy = rng.random((3, config.NUM_KEYPOINTS, 2)).astype(np.float32)

    torch_sam = build_sam_from_keypoints(torch.from_numpy(xy)).numpy()
    numpy_sam = np.stack([generate_sam(sample) for sample in xy])

    assert np.allclose(torch_sam, numpy_sam, atol=1e-6)

    # Round trip through the diagonal.
    diag = sam_diagonal(torch.from_numpy(torch_sam)).numpy()
    assert np.allclose(diag, xy, atol=1e-6)


def test_root_relative_reconstruction_and_loss():
    original = {
        "POSE_REPRESENTATION": config.POSE_REPRESENTATION,
        "LAMBDA_BONE": config.LAMBDA_BONE,
        "LAMBDA_ANGLE": config.LAMBDA_ANGLE,
    }
    config.POSE_REPRESENTATION = "root_relative"
    config.LAMBDA_BONE = 0.1
    config.LAMBDA_ANGLE = 0.1

    try:
        rng = np.random.default_rng(1)
        b, k = 4, config.NUM_KEYPOINTS
        target = torch.from_numpy(
            rng.random((b, k, 2)).astype(np.float32)
        )

        root, scale = pose_root_and_scale(target)
        relative = (target - root.unsqueeze(1)) / scale.unsqueeze(-1)

        predicted_sam = build_sam_from_keypoints(relative)
        root_vector = torch.cat([root, torch.log(scale)], dim=1)

        reconstructed = reconstruct_absolute_keypoints(
            predicted_sam,
            root_vector,
        )
        assert torch.allclose(reconstructed, target, atol=1e-5)

        # A perfect prediction must give (near) zero keypoint losses.
        criterion = ConfidenceWeightedMultiTaskLoss()
        confidence = torch.ones(b, k)
        target_sam = build_sam_from_keypoints(target)
        pose_class = torch.zeros(b, dtype=torch.long)

        losses = criterion(
            predicted_sam,
            target_sam,
            target,
            confidence,
            pose_logits=None,
            pose_class=pose_class,
            root=root_vector,
        )

        for key in ("coordinate", "relpose", "root", "bone", "angle"):
            assert losses[key].item() < 1e-8, (key, losses[key].item())

        # Gradients must flow through every keypoint term.
        predicted_sam = build_sam_from_keypoints(
            relative + 0.05 * torch.randn_like(relative)
        ).requires_grad_(True)
        root_param = root_vector.clone().requires_grad_(True)

        losses = criterion(
            predicted_sam,
            target_sam,
            target,
            confidence,
            pose_logits=None,
            pose_class=pose_class,
            root=root_param,
        )
        losses["total"].backward()
        assert predicted_sam.grad is not None
        assert root_param.grad is not None
        assert math.isfinite(losses["total"].item())
    finally:
        for key, value in original.items():
            setattr(config, key, value)


def test_multitask_model_variants():
    original = {
        "POSE_REPRESENTATION": config.POSE_REPRESENTATION,
        "ADVERSARIAL_SUBJECT": config.ADVERSARIAL_SUBJECT,
        "SAM_DECODER": config.SAM_DECODER,
        "MODEL_NAME": config.MODEL_NAME,
    }
    config.MODEL_NAME = "cnn_gru_multitask_sam"
    config.POSE_REPRESENTATION = "root_relative"
    config.ADVERSARIAL_SUBJECT = True
    config.SAM_DECODER = "gcn"

    try:
        model = build_model()
        x = torch.randn(
            2,
            config.BURSTS_PER_WINDOW,
            config.SNAPSHOTS_PER_BURST,
            config.NUM_RX,
            config.NUM_USABLE_SUBCARRIERS,
        )
        output = model(x)

        assert output["sam"].shape == (
            2,
            config.SAM_CHANNELS,
            config.NUM_KEYPOINTS,
            config.NUM_KEYPOINTS,
        )
        assert output["pose_logits"].shape == (2, config.NUM_POSE_CLASSES)
        assert output["root"].shape == (2, 3)
        assert output["subject_logits"].shape == (2, len(config.SUBJECT_IDS))

        # Gradient reversal: the subject loss must produce gradients in the
        # backbone with reversed sign relative to a normal head. Just check
        # gradients exist and are finite.
        subject_loss = output["subject_logits"].sum()
        subject_loss.backward()
        for name, parameter in model.backbone.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
    finally:
        for key, value in original.items():
            setattr(config, key, value)


if __name__ == "__main__":
    test_geometry_functions()
    test_build_sam_matches_numpy()
    test_root_relative_reconstruction_and_loss()
    test_multitask_model_variants()
    print("Experiment feature tests passed.")
