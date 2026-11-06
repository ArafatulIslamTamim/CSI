import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import CSIPoseDataset, load_split_indices
from .losses import (
    ConfidenceWeightedMultiTaskLoss,
    reconstruct_absolute_keypoints,
)
from .metrics import mpjpe, normalized_pck, pixel_mpjpe, pixel_pck
from .models import build_model, unpack_model_output


def print_split_distribution():
    data = np.load(config.DATASET_FILE)
    train_idx, val_idx, test_idx = load_split_indices()

    print("\nSPLIT DISTRIBUTION")
    print("==================")

    for name, idx in [("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)]:
        subjects = data["subject_id"][idx].astype(str)
        poses = data["pose_name"][idx].astype(str)
        print(f"\n{name}")
        print("-" * 30)
        print("samples :", len(idx))
        print("subjects:", sorted(np.unique(subjects).tolist()))
        for pose in sorted(np.unique(poses).tolist()):
            print(f"{pose:10s}: {int(np.sum(poses == pose))}")


def evaluate_subset(model, device, global_indices, pose_name):
    data = np.load(config.DATASET_FILE)
    all_pose_names = data["pose_name"].astype(str)

    subset_indices = np.asarray(
        [
            int(i)
            for i in global_indices
            if all_pose_names[int(i)].lower() == pose_name.lower()
        ],
        dtype=np.int64,
    )

    if len(subset_indices) == 0:
        return None

    ds = CSIPoseDataset(subset_indices)
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    criterion = ConfidenceWeightedMultiTaskLoss()
    predicted = []
    ground_truth = []
    widths = []
    heights = []

    loss_sums = None

    class_correct = 0
    class_count = 0
    sample_count = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            csi = batch["csi"].to(device=device, dtype=torch.float32)
            target_sam = batch["sam"].to(device=device, dtype=torch.float32)
            target_xy = batch["keypoints"].to(device=device, dtype=torch.float32)
            confidence = batch["confidence"].to(device=device, dtype=torch.float32)
            pose_class = batch["pose_class"].to(device=device, dtype=torch.long)
            subject_class = batch["subject_class"].to(device=device, dtype=torch.long)

            output = model(csi)
            pred_sam, pose_logits = unpack_model_output(output)
            root = output.get("root") if isinstance(output, dict) else None
            subject_logits = (
                output.get("subject_logits") if isinstance(output, dict) else None
            )

            losses = criterion(
                pred_sam,
                target_sam,
                target_xy,
                confidence,
                pose_logits=pose_logits,
                pose_class=pose_class,
                root=root,
                subject_logits=subject_logits,
                subject_class=subject_class,
            )

            bs = csi.shape[0]
            sample_count += bs
            if loss_sums is None:
                loss_sums = {key: 0.0 for key in losses}
            for key in loss_sums:
                loss_sums[key] += float(losses[key].item()) * bs

            if pose_logits is not None:
                cls_pred = pose_logits.argmax(dim=1)
                class_correct += int((cls_pred == pose_class).sum().item())
                class_count += bs

            predicted.append(
                reconstruct_absolute_keypoints(pred_sam, root).cpu().numpy()
            )
            ground_truth.append(target_xy.cpu().numpy())
            widths.extend(batch["frame_width"].numpy().tolist())
            heights.extend(batch["frame_height"].numpy().tolist())

    predicted = np.concatenate(predicted, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    widths = np.asarray(widths, dtype=np.float32)
    heights = np.asarray(heights, dtype=np.float32)

    # Empty-room samples have no keypoint ground truth, so keypoint
    # metrics do not apply; only classification is meaningful.
    is_empty_pose = (
        pose_name.lower() == config.EMPTY_POSE_NAME
    )

    if is_empty_pose:
        overall_pck = None
        per_joint_pck = None
        pixel_results = None
        overall_mpjpe = None
        per_joint_mpjpe = None
        mpjpe_px = None
    else:
        overall_pck, per_joint_pck = normalized_pck(
            predicted,
            ground_truth,
            config.PCK_ALPHA,
        )

        pixel_results = pixel_pck(
            predicted,
            ground_truth,
            widths,
            heights,
            config.PIXEL_PCK_THRESHOLDS,
        )

        overall_mpjpe, per_joint_mpjpe = mpjpe(
            predicted,
            ground_truth,
        )

        mpjpe_px = pixel_mpjpe(
            predicted,
            ground_truth,
            widths,
            heights,
        )

    denom = max(sample_count, 1)

    return {
        "pose": pose_name,
        "samples": int(sample_count),
        "combined_loss": loss_sums["total"] / denom,
        "coordinate_loss": loss_sums["coordinate"] / denom,
        "relative_sam_loss": loss_sums["relative"] / denom,
        "classification_loss": loss_sums["classification"] / denom,
        "pose_classification_accuracy": (
            class_correct / class_count
            if class_count > 0
            else None
        ),
        "normalized_pck_alpha": config.PCK_ALPHA,
        "normalized_pck": (
            None if overall_pck is None else float(overall_pck)
        ),
        "normalized_pck_per_joint": (
            None
            if per_joint_pck is None
            else [float(v) for v in per_joint_pck]
        ),
        "normalized_mpjpe": (
            None if overall_mpjpe is None else float(overall_mpjpe)
        ),
        "normalized_mpjpe_per_joint": (
            None
            if per_joint_mpjpe is None
            else [float(v) for v in per_joint_mpjpe]
        ),
        "pixel_mpjpe": (
            None if mpjpe_px is None else float(mpjpe_px)
        ),
        "pixel_pck": (
            None
            if pixel_results is None
            else {k: float(v) for k, v in pixel_results.items()}
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    args = parser.parse_args()

    config.ensure_directories()

    if not config.BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Best checkpoint missing: {config.BEST_MODEL_FILE}")

    print_split_distribution()

    _, val_idx, test_idx = load_split_indices()
    selected_idx = val_idx if args.split == "val" else test_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    checkpoint = torch.load(config.BEST_MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nPER-POSE {args.split.upper()} EVALUATION")
    print("=" * 34)
    print("model:", config.MODEL_NAME)
    print("device:", device)
    print("checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    results = {}
    for pose in config.POSE_CLASSES:
        result = evaluate_subset(model, device, selected_idx, pose)
        if result is None:
            continue
        results[pose] = result

        print(f"\n{pose.upper()}")
        print("-" * 30)
        print("samples:", result["samples"])
        if result["normalized_pck"] is not None:
            print(
                f"PCK@{config.PCK_ALPHA}: "
                f"{result['normalized_pck']:.4f} "
                f"({result['normalized_pck']*100:.2f}%)"
            )
        if result["pose_classification_accuracy"] is not None:
            a = result["pose_classification_accuracy"]
            print(f"classification accuracy: {a:.4f} ({a*100:.2f}%)")
        if result["pixel_pck"] is not None:
            for key, value in result["pixel_pck"].items():
                print(f"{key}: {value:.4f} ({value*100:.2f}%)")
        print(f"coordinate loss: {result['coordinate_loss']:.6f}")
        print(f"relative SAM loss: {result['relative_sam_loss']:.6f}")

    out_path = (
        config.EVALUATION_DIR
        / (
            f"{config.MODEL_NAME}_{config.EXPERIMENT_TAG}"
            f"_{args.split}_metrics_by_pose.json"
        )
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": config.MODEL_NAME,
                "split": args.split,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
