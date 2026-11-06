import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import (
    CSIPoseDataset,
    compute_adaptive_stats,
    load_split_indices,
)
from .losses import (
    ConfidenceWeightedMultiTaskLoss,
    reconstruct_absolute_keypoints,
)
from .metrics import mpjpe, normalized_pck, pixel_mpjpe, pixel_pck
from .models import build_model, count_trainable_parameters, unpack_model_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapt-norm",
        action="store_true",
        help=(
            "AdaBN-style test-time adaptation: recompute the global CSI "
            "normalization statistics from the unlabeled test-split CSI "
            "instead of the training split. Outputs get an _adaptnorm suffix."
        ),
    )
    args = parser.parse_args()

    config.ensure_directories()

    if not config.BEST_MODEL_FILE.exists():
        raise FileNotFoundError("Best model missing. Train first.")

    _, _, test_idx = load_split_indices()
    test_ds = CSIPoseDataset(test_idx)

    if args.adapt_norm:
        adapted_mean, adapted_std = compute_adaptive_stats(test_idx)
        test_ds.mean = adapted_mean
        test_ds.std = adapted_std
        print("[adapt-norm] using test-split CSI normalization statistics")

    loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    checkpoint = torch.load(config.BEST_MODEL_FILE, map_location=device)
    if checkpoint.get("model_name") != config.MODEL_NAME:
        raise RuntimeError("Checkpoint/model config mismatch.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = ConfidenceWeightedMultiTaskLoss()

    predicted = []
    ground_truth = []
    confidence_all = []
    widths = []
    heights = []
    global_indices = []
    predicted_classes = []
    target_classes = []

    sums = None
    count = 0

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
            count += bs
            if sums is None:
                sums = {key: 0.0 for key in losses}
            for key in sums:
                sums[key] += float(losses[key].item()) * bs

            pred_xy = reconstruct_absolute_keypoints(pred_sam, root).cpu().numpy()
            predicted.append(pred_xy)
            ground_truth.append(target_xy.cpu().numpy())
            confidence_all.append(confidence.cpu().numpy())
            widths.extend(batch["frame_width"].numpy().tolist())
            heights.extend(batch["frame_height"].numpy().tolist())
            global_indices.extend(batch["global_index"].numpy().tolist())

            target_classes.extend(pose_class.cpu().numpy().tolist())

            if pose_logits is not None:
                predicted_classes.extend(
                    pose_logits.argmax(dim=1).cpu().numpy().tolist()
                )

    predicted = np.concatenate(predicted, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    confidence_all = np.concatenate(confidence_all, axis=0)
    widths = np.asarray(widths, dtype=np.float32)
    heights = np.asarray(heights, dtype=np.float32)
    target_classes_np = np.asarray(target_classes, dtype=np.int64)

    # Empty-room samples have no keypoint ground truth: keypoint metrics
    # are computed on non-empty samples only.
    if config.EMPTY_CLASS_INDEX < 0:
        has_pose = np.ones(
            target_classes_np.shape,
            dtype=bool,
        )
    else:
        has_pose = target_classes_np != config.EMPTY_CLASS_INDEX
    n_with_pose = int(has_pose.sum())

    if n_with_pose > 0:
        overall_pck, per_joint = normalized_pck(
            predicted[has_pose],
            ground_truth[has_pose],
            config.PCK_ALPHA,
        )

        pixel_results = pixel_pck(
            predicted[has_pose],
            ground_truth[has_pose],
            widths[has_pose],
            heights[has_pose],
            config.PIXEL_PCK_THRESHOLDS,
        )

        overall_mpjpe, per_joint_mpjpe = mpjpe(
            predicted[has_pose],
            ground_truth[has_pose],
        )

        mpjpe_px = pixel_mpjpe(
            predicted[has_pose],
            ground_truth[has_pose],
            widths[has_pose],
            heights[has_pose],
        )
    else:
        overall_pck = float("nan")
        per_joint = np.full(config.NUM_KEYPOINTS, np.nan, dtype=np.float32)
        pixel_results = {
            f"PCK@{int(t)}": float("nan")
            for t in config.PIXEL_PCK_THRESHOLDS
        }
        overall_mpjpe = float("nan")
        per_joint_mpjpe = np.full(config.NUM_KEYPOINTS, np.nan, dtype=np.float32)
        mpjpe_px = float("nan")

    denominator = max(count, 1)

    if predicted_classes:
        predicted_classes_np = np.asarray(predicted_classes, dtype=np.int64)

        classification_accuracy = float(
            np.mean(predicted_classes_np == target_classes_np)
        )

        n_classes = config.NUM_POSE_CLASSES
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(target_classes_np, predicted_classes_np):
            confusion[t, p] += 1

        per_class = {}
        for class_idx, class_name in enumerate(config.POSE_CLASSES):
            tp = int(confusion[class_idx, class_idx])
            predicted_positive = int(confusion[:, class_idx].sum())
            actual_positive = int(confusion[class_idx, :].sum())

            per_class[class_name] = {
                "support": actual_positive,
                "precision": (
                    tp / predicted_positive
                    if predicted_positive > 0
                    else None
                ),
                "recall": (
                    tp / actual_positive
                    if actual_positive > 0
                    else None
                ),
            }

        classification_report = {
            "classes": list(config.POSE_CLASSES),
            "confusion_matrix": confusion.tolist(),
            "per_class": per_class,
            "empty_precision": (
                per_class[config.EMPTY_POSE_NAME]["precision"]
                if config.EMPTY_POSE_NAME in per_class
                else None
            ),
            "empty_recall": (
                per_class[config.EMPTY_POSE_NAME]["recall"]
                if config.EMPTY_POSE_NAME in per_class
                else None
            ),
        }
    else:
        classification_accuracy = None
        classification_report = None

    metrics = {
        "model": config.MODEL_NAME,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "adapt_norm": bool(args.adapt_norm),
        "pose_representation": config.POSE_REPRESENTATION,
        "test_samples": int(count),
        "test_samples_with_pose": n_with_pose,
        "test_samples_empty": int(count) - n_with_pose,
        "combined_loss": sums["total"] / denominator,
        "coordinate_loss": sums["coordinate"] / denominator,
        "relative_sam_loss": sums["relative"] / denominator,
        "classification_loss": sums["classification"] / denominator,
        "loss_terms": {
            key: value / denominator for key, value in sums.items()
        },
        "pose_classification_accuracy": classification_accuracy,
        "classification_report": classification_report,
        "normalized_pck_alpha": config.PCK_ALPHA,
        "normalized_pck": float(overall_pck),
        "normalized_pck_per_joint": [float(v) for v in per_joint],
        "normalized_mpjpe": float(overall_mpjpe),
        "normalized_mpjpe_per_joint": [float(v) for v in per_joint_mpjpe],
        "pixel_mpjpe": float(mpjpe_px),
        "pixel_pck": {k: float(v) for k, v in pixel_results.items()},
        "trainable_parameters": count_trainable_parameters(model),
    }

    metrics_file = config.METRICS_FILE
    predictions_file = config.TEST_PREDICTIONS_FILE

    if args.adapt_norm:
        metrics_file = metrics_file.with_name(
            metrics_file.name.replace("_metrics.json", "_adaptnorm_metrics.json")
        )
        predictions_file = predictions_file.with_name(
            predictions_file.name.replace(
                "_test_predictions.npz",
                "_adaptnorm_test_predictions.npz",
            )
        )

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        predictions_file,
        global_index=np.asarray(global_indices, dtype=np.int64),
        predicted_keypoints=predicted,
        ground_truth_keypoints=ground_truth,
        confidence=confidence_all,
        predicted_pose_class=np.asarray(predicted_classes, dtype=np.int64),
        target_pose_class=np.asarray(target_classes, dtype=np.int64),
    )

    print("\nEVALUATION")
    print("==========")
    print(f"model: {config.MODEL_NAME}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"test samples: {count} (with pose: {n_with_pose}, empty: {count - n_with_pose})")
    print(
        f"normalized PCK@{config.PCK_ALPHA} (non-empty only): "
        f"{overall_pck:.4f} ({overall_pck*100:.2f}%)"
    )
    print(
        f"normalized MPJPE (non-empty only): {overall_mpjpe:.4f} | "
        f"pixel MPJPE: {mpjpe_px:.2f}px"
    )
    if classification_accuracy is not None:
        print(
            "pose classification accuracy: "
            f"{classification_accuracy:.4f} "
            f"({classification_accuracy*100:.2f}%)"
        )
    if classification_report is not None:
        print(f"confusion matrix rows=true cols=pred {list(config.POSE_CLASSES)}:")
        for row in classification_report["confusion_matrix"]:
            print(f"  {row}")
        for class_name, stats in classification_report["per_class"].items():
            precision = stats["precision"]
            recall = stats["recall"]
            print(
                f"  {class_name:10s} "
                f"precision={precision if precision is None else f'{precision:.4f}'} "
                f"recall={recall if recall is None else f'{recall:.4f}'} "
                f"support={stats['support']}"
            )
    for name, value in pixel_results.items():
        print(f"{name}: {value:.4f} ({value*100:.2f}%)")
    print(f"metrics: {metrics_file}")


if __name__ == "__main__":
    main()
