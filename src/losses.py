import torch
from torch import nn

from . import config
from .geometry import (
    ANGLE_TRIPLETS,
    BONES,
    bone_lengths,
    build_sam_from_keypoints,
    joint_cosines,
)

_BONE_A = [a for a, _ in BONES]
_BONE_B = [b for _, b in BONES]
_TRIPLET_A = [t[0] for t in ANGLE_TRIPLETS]
_TRIPLET_J = [t[1] for t in ANGLE_TRIPLETS]
_TRIPLET_C = [t[2] for t in ANGLE_TRIPLETS]


def sam_diagonal(sam):
    """[B,2,K,K] -> [B,K,2]."""
    x = torch.diagonal(sam[:, 0], dim1=1, dim2=2)
    y = torch.diagonal(sam[:, 1], dim1=1, dim2=2)
    return torch.stack([x, y], dim=-1)


def pose_root_and_scale(keypoints_xy):
    """
    Root = mid-hip, scale = torso length (mid-shoulder to root),
    floored at MIN_BODY_SCALE.

    [B,K,2] -> (root [B,2], scale [B,1])
    """
    root = keypoints_xy[:, list(config.ROOT_JOINT_INDICES)].mean(dim=1)
    shoulder = keypoints_xy[:, list(config.SHOULDER_JOINT_INDICES)].mean(dim=1)

    scale = (
        (shoulder - root)
        .norm(dim=-1, keepdim=True)
        .clamp_min(config.MIN_BODY_SCALE)
    )

    return root, scale


def reconstruct_absolute_keypoints(predicted_sam, root=None):
    """
    Absolute normalized keypoints from a model prediction.

    In "absolute" mode the SAM diagonal already holds absolute coordinates.
    In "root_relative" mode the diagonal holds (xy - root)/scale and `root`
    is [B,3] = (root_x, root_y, log_scale).
    """
    diagonal = sam_diagonal(predicted_sam)

    if root is None or config.POSE_REPRESENTATION == "absolute":
        return diagonal

    root_xy = root[:, :2].unsqueeze(1)                # [B,1,2]
    scale = torch.exp(root[:, 2]).view(-1, 1, 1)      # [B,1,1]

    return root_xy + scale * diagonal


class ConfidenceWeightedMultiTaskLoss(nn.Module):
    """
    L = lambda_coord    * L_absolute_coordinate
      + lambda_relative * L_relative_SAM (off-diagonal pairwise)
      + lambda_class    * L_pose_classification
      + lambda_relpose  * L_root_relative_coordinate   (root_relative mode)
      + lambda_root     * L_root_and_scale             (root_relative mode)
      + lambda_bone     * L_bone_length                (if enabled)
      + lambda_angle    * L_joint_angle                (if enabled)
      + lambda_subject  * L_subject_adversarial        (if enabled)

    YOLO-Pose keypoint confidence weights all keypoint-derived terms.
    Empty-room samples contribute only through classification (and the
    subject term excludes them as well).
    """

    def __init__(self):
        super().__init__()
        # Built per-forward when CLASS_LOSS_WEIGHTS is set so weights
        # stay on the same device as logits.
        self.classification = nn.CrossEntropyLoss()

    def _classification_loss(self, pose_logits, pose_class):
        weights = getattr(config, "CLASS_LOSS_WEIGHTS", None)
        if weights is None:
            return self.classification(pose_logits, pose_class)

        weight = torch.as_tensor(
            weights,
            device=pose_logits.device,
            dtype=pose_logits.dtype,
        )
        if weight.numel() != pose_logits.shape[-1]:
            raise ValueError(
                f"CLASS_LOSS_WEIGHTS length {weight.numel()} != "
                f"num classes {pose_logits.shape[-1]}"
            )
        return nn.functional.cross_entropy(
            pose_logits,
            pose_class,
            weight=weight,
        )

    def forward(
        self,
        predicted_sam,
        target_sam,
        target_keypoints,
        confidence,
        pose_logits=None,
        pose_class=None,
        root=None,
        subject_logits=None,
        subject_class=None,
    ):
        root_relative = (
            config.POSE_REPRESENTATION == "root_relative"
            and root is not None
        )

        joint_weight = torch.clamp(
            confidence,
            min=config.MIN_CONFIDENCE_WEIGHT,
            max=1.0,
        )

        # Empty-room samples carry no keypoint supervision. Zero their
        # coordinate/SAM weights entirely so the clamp floor above cannot
        # pull predictions toward the dummy zero targets; such samples
        # contribute only through the classification loss.
        if pose_class is not None and config.EMPTY_CLASS_INDEX >= 0:
            supervised = pose_class != config.EMPTY_CLASS_INDEX
        else:
            supervised = confidence.sum(dim=1) > 0

        supervised_f = supervised.to(joint_weight.dtype)
        joint_weight = joint_weight * supervised_f[:, None]

        zero = predicted_sam.new_zeros(())

        # ---------- representation decomposition ----------
        predicted_diagonal = sam_diagonal(predicted_sam)

        if root_relative:
            target_root, target_scale = pose_root_and_scale(target_keypoints)
            target_relative = (
                target_keypoints - target_root.unsqueeze(1)
            ) / target_scale.unsqueeze(-1)
            relative_target_sam = build_sam_from_keypoints(target_relative)

            predicted_absolute = reconstruct_absolute_keypoints(
                predicted_sam,
                root,
            )
            sam_loss_target = relative_target_sam
        else:
            predicted_absolute = predicted_diagonal
            sam_loss_target = target_sam

        # ---------- absolute coordinate loss ----------
        coord_error = (predicted_absolute - target_keypoints) ** 2
        coord_weight = joint_weight[..., None]

        coord_loss = (
            (coord_error * coord_weight).sum()
            /
            (coord_weight.sum() * 2.0).clamp_min(1e-8)
        )

        # ---------- root-relative coordinate + root/scale losses ----------
        if root_relative:
            relpose_error = (predicted_diagonal - target_relative) ** 2
            relpose_loss = (
                (relpose_error * coord_weight).sum()
                /
                (coord_weight.sum() * 2.0).clamp_min(1e-8)
            )

            root_error = (
                (root[:, :2] - target_root) ** 2
            ).sum(dim=1) + (
                root[:, 2] - torch.log(target_scale.squeeze(1))
            ) ** 2

            root_loss = (
                (root_error * supervised_f).sum()
                / supervised_f.sum().clamp_min(1e-8)
            )
        else:
            relpose_loss = zero
            root_loss = zero

        # ---------- pairwise structural SAM loss (off-diagonal only) ----------
        k = predicted_sam.shape[-1]
        pair_weight = joint_weight[:, :, None] * joint_weight[:, None, :]

        offdiag = (
            1.0
            - torch.eye(
                k,
                device=predicted_sam.device,
                dtype=predicted_sam.dtype,
            )
        )
        pair_weight = pair_weight * offdiag[None]

        relative_error = (predicted_sam - sam_loss_target) ** 2
        relative_weight = pair_weight[:, None]

        relative_loss = (
            (relative_error * relative_weight).sum()
            /
            (
                relative_weight.sum()
                * predicted_sam.shape[1]
            ).clamp_min(1e-8)
        )

        # ---------- geometry losses on absolute coordinates ----------
        if config.LAMBDA_BONE > 0.0:
            predicted_bones = bone_lengths(predicted_absolute)
            target_bones = bone_lengths(target_keypoints)

            bone_weight = torch.minimum(
                joint_weight[:, _BONE_A],
                joint_weight[:, _BONE_B],
            )

            bone_loss = (
                ((predicted_bones - target_bones).abs() * bone_weight).sum()
                / bone_weight.sum().clamp_min(1e-8)
            )
        else:
            bone_loss = zero

        if config.LAMBDA_ANGLE > 0.0:
            predicted_cos = joint_cosines(predicted_absolute)
            target_cos = joint_cosines(target_keypoints)

            triplet_weight = torch.minimum(
                torch.minimum(
                    joint_weight[:, _TRIPLET_A],
                    joint_weight[:, _TRIPLET_J],
                ),
                joint_weight[:, _TRIPLET_C],
            )

            angle_loss = (
                ((predicted_cos - target_cos) ** 2 * triplet_weight).sum()
                / triplet_weight.sum().clamp_min(1e-8)
            )
        else:
            angle_loss = zero

        # ---------- classification ----------
        if pose_logits is not None:
            if pose_class is None:
                raise ValueError("pose_class target is required with pose_logits")
            class_loss = self._classification_loss(pose_logits, pose_class)
        else:
            class_loss = zero

        # ---------- subject-adversarial (gradient already reversed in model) ----------
        if subject_logits is not None and subject_class is not None:
            valid = subject_class >= 0
            if valid.any():
                subject_loss = self.classification(
                    subject_logits[valid],
                    subject_class[valid],
                )
            else:
                subject_loss = zero
        else:
            subject_loss = zero

        total = (
            config.LAMBDA_COORD * coord_loss
            + config.LAMBDA_RELATIVE * relative_loss
            + config.LAMBDA_CLASS * class_loss
            + config.LAMBDA_RELPOSE * relpose_loss
            + config.LAMBDA_ROOT * root_loss
            + config.LAMBDA_BONE * bone_loss
            + config.LAMBDA_ANGLE * angle_loss
            + config.LAMBDA_SUBJECT * subject_loss
        )

        return {
            "total": total,
            "coordinate": coord_loss,
            "relative": relative_loss,
            "classification": class_loss,
            "relpose": relpose_loss,
            "root": root_loss,
            "bone": bone_loss,
            "angle": angle_loss,
            "subject": subject_loss,
        }


def bbox_root_and_scale(keypoints_xy):
    """
    Root = bbox center; scale = sqrt(w^2 + h^2) in normalized coords.
    [B,K,2] -> (root [B,2], scale [B,1])
    """
    xmin = keypoints_xy.min(dim=1).values[:, 0:1]
    xmax = keypoints_xy.max(dim=1).values[:, 0:1]
    ymin = keypoints_xy.min(dim=1).values[:, 1:2]
    ymax = keypoints_xy.max(dim=1).values[:, 1:2]
    root_x = (xmin + xmax) * 0.5
    root_y = (ymin + ymax) * 0.5
    root = torch.cat([root_x, root_y], dim=1)
    w = (xmax - xmin).clamp_min(1e-6)
    h = (ymax - ymin).clamp_min(1e-6)
    scale = torch.sqrt(w * w + h * h).clamp_min(config.MIN_BODY_SCALE)
    return root, scale


class ThreeHeadV2Loss(nn.Module):
    """Presence + posture + root-relative pose (v2 TCN)."""

    def forward(
        self,
        output,
        target_keypoints,
        confidence,
        *,
        pose_name_is_empty,
        posture_class=None,
    ):
        zero = output["presence_logit"].new_zeros(())
        occupied = ~pose_name_is_empty
        occ_f = occupied.to(confidence.dtype)

        # Presence BCE
        presence_target = occupied.to(output["presence_logit"].dtype)
        presence_loss = nn.functional.binary_cross_entropy_with_logits(
            output["presence_logit"].squeeze(-1),
            presence_target,
        )

        # Posture CE (occupied only)
        if posture_class is not None and occupied.any():
            posture_loss = nn.functional.cross_entropy(
                output["posture_logits"][occupied],
                posture_class[occupied],
            )
        else:
            posture_loss = zero

        joint_weight = torch.clamp(
            confidence,
            min=config.MIN_CONFIDENCE_WEIGHT,
            max=1.0,
        )
        joint_weight = joint_weight * occ_f[:, None]

        if occupied.any():
            tgt_root, tgt_scale = bbox_root_and_scale(target_keypoints)
            tgt_rel = (
                target_keypoints - tgt_root.unsqueeze(1)
            ) / tgt_scale.unsqueeze(-1)

            pred_root = output["root_scale"][:, :2]
            pred_scale = output["root_scale"][:, 2:3].clamp_min(
                config.MIN_BODY_SCALE
            )
            pred_rel = output["rel_pose"]

            rel_err = nn.functional.smooth_l1_loss(
                pred_rel,
                tgt_rel,
                reduction="none",
            )
            pose_loss = (
                (rel_err * joint_weight.unsqueeze(-1)).sum()
                / (joint_weight.sum() * 2.0).clamp_min(1e-8)
            )

            root_loss = nn.functional.smooth_l1_loss(
                pred_root[occupied],
                tgt_root[occupied],
                reduction="mean",
            )
            scale_loss = nn.functional.smooth_l1_loss(
                pred_scale[occupied],
                tgt_scale[occupied],
                reduction="mean",
            )

            pred_abs = (
                pred_root.unsqueeze(1)
                + pred_scale.unsqueeze(-1) * pred_rel
            )
            if config.LAMBDA_BONE > 0.0:
                pred_bones = bone_lengths(pred_abs)
                tgt_bones = bone_lengths(target_keypoints)
                bone_w = torch.minimum(
                    joint_weight[:, _BONE_A],
                    joint_weight[:, _BONE_B],
                )
                bone_loss = (
                    ((pred_bones - tgt_bones).abs() * bone_w).sum()
                    / bone_w.sum().clamp_min(1e-8)
                )
            else:
                bone_loss = zero
        else:
            pose_loss = zero
            root_loss = zero
            scale_loss = zero
            bone_loss = zero

        total = (
            config.LAMBDA_PRESENCE * presence_loss
            + config.LAMBDA_POSTURE * posture_loss
            + config.LAMBDA_POSE * pose_loss
            + config.LAMBDA_ROOT * root_loss
            + config.LAMBDA_SCALE * scale_loss
            + config.LAMBDA_BONE * bone_loss
        )

        return {
            "total": total,
            "presence": presence_loss,
            "posture": posture_loss,
            "pose": pose_loss,
            "root": root_loss,
            "scale": scale_loss,
            "bone": bone_loss,
        }


# Backward-compatible alias for any old imports.
ConfidenceWeightedSAMLoss = ConfidenceWeightedMultiTaskLoss
