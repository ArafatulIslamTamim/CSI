"""
Differentiable human-body geometry for COCO-17 keypoints.

Used by the geometry loss terms (bone lengths, joint angles) and by the
GCN decoder (skeleton adjacency).
"""

import torch

# COCO-17 skeleton edges (matches visualize.SKELETON).
BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)

# (endpoint, joint, endpoint): the angle is measured at the middle index.
# Elbows, knees, and hips: the joints that distinguish sitting from standing.
ANGLE_TRIPLETS = (
    (5, 7, 9),      # left elbow
    (6, 8, 10),     # right elbow
    (11, 13, 15),   # left knee
    (12, 14, 16),   # right knee
    (5, 11, 13),    # left hip
    (6, 12, 14),    # right hip
)

_BONE_A = tuple(a for a, _ in BONES)
_BONE_B = tuple(b for _, b in BONES)
_TRIPLET_A = tuple(t[0] for t in ANGLE_TRIPLETS)
_TRIPLET_J = tuple(t[1] for t in ANGLE_TRIPLETS)
_TRIPLET_C = tuple(t[2] for t in ANGLE_TRIPLETS)


def bone_lengths(keypoints_xy):
    """[B,K,2] -> [B,len(BONES)] euclidean bone lengths."""
    a = keypoints_xy[:, _BONE_A]
    b = keypoints_xy[:, _BONE_B]
    return (a - b).norm(dim=-1)


def joint_cosines(keypoints_xy, eps=1e-6):
    """
    [B,K,2] -> [B,len(ANGLE_TRIPLETS)] cosine of the angle at each middle
    joint. Straight limb -> -1, right angle -> 0.
    """
    a = keypoints_xy[:, _TRIPLET_A]
    j = keypoints_xy[:, _TRIPLET_J]
    c = keypoints_xy[:, _TRIPLET_C]

    v1 = a - j
    v2 = c - j

    denominator = (
        v1.norm(dim=-1) * v2.norm(dim=-1)
    ).clamp_min(eps)

    return (v1 * v2).sum(dim=-1) / denominator


def build_sam_from_keypoints(keypoints_xy):
    """
    [B,K,2] -> [B,2,K,K] SAM tensor:
    diagonal = coordinates, off-diagonal = pairwise coordinate differences.
    Torch counterpart of sam.generate_sam.
    """
    x = keypoints_xy[..., 0]
    y = keypoints_xy[..., 1]

    gx = x[:, :, None] - x[:, None, :]
    gy = y[:, :, None] - y[:, None, :]

    k = keypoints_xy.shape[1]
    idx = torch.arange(k, device=keypoints_xy.device)

    gx = gx.clone()
    gy = gy.clone()
    gx[:, idx, idx] = x
    gy[:, idx, idx] = y

    return torch.stack([gx, gy], dim=1)


def skeleton_adjacency(num_keypoints, device=None):
    """
    Symmetrically normalized COCO skeleton adjacency with self-loops:
    D^{-1/2} (A + I) D^{-1/2}. Returns [K,K].
    """
    adjacency = torch.eye(num_keypoints, device=device)
    for a, b in BONES:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    degree = adjacency.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)

    return adjacency * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
