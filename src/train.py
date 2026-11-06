import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import (
    CSIPoseDataset,
    compute_and_save_csi_stats,
    create_split_indices,
)
from .losses import ConfidenceWeightedMultiTaskLoss, ThreeHeadV2Loss
from .models import (
    build_model,
    count_trainable_parameters,
    unpack_model_output,
)
from .samplers import SessionBalancedSampler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l1_regularization(model):
    """Sum of absolute trainable weights (for L1 penalty)."""
    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        term = param.abs().sum()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return total


def _is_v2_model():
    return config.MODEL_NAME == "tcn_three_head_v2"


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    sums = None

    class_correct = 0
    class_count = 0
    count = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            csi = batch["csi"].to(device=device, dtype=torch.float32)
            target_sam = batch["sam"].to(device=device, dtype=torch.float32)
            target_xy = batch["keypoints"].to(device=device, dtype=torch.float32)
            confidence = batch["confidence"].to(device=device, dtype=torch.float32)
            pose_class = batch["pose_class"].to(device=device, dtype=torch.long)
            subject_class = batch["subject_class"].to(device=device, dtype=torch.long)

            if training:
                optimizer.zero_grad(set_to_none=True)

            if _is_v2_model():
                output = model(csi)
                loss_dict = criterion(
                    output,
                    target_xy,
                    confidence,
                    pose_name_is_empty=batch["is_empty"].to(device),
                    posture_class=batch["posture_class"].to(device),
                )
                pose_logits = output.get("posture_logits")
                pose_class = batch["posture_class"].to(device)
            else:
                output = model(csi)
                predicted_sam, pose_logits = unpack_model_output(output)
                root = output.get("root") if isinstance(output, dict) else None
                subject_logits = (
                    output.get("subject_logits") if isinstance(output, dict) else None
                )
                loss_dict = criterion(
                    predicted_sam,
                    target_sam,
                    target_xy,
                    confidence,
                    pose_logits=pose_logits,
                    pose_class=pose_class,
                    root=root,
                    subject_logits=subject_logits,
                    subject_class=subject_class,
                )

            # L1 on weights (training only). L2 is applied via AdamW weight_decay.
            l1_term = loss_dict["total"].new_zeros(())
            if training and config.L1_WEIGHT > 0:
                l1_term = config.L1_WEIGHT * l1_regularization(model)
                loss_dict = dict(loss_dict)
                loss_dict["l1"] = l1_term
                loss_dict["total"] = loss_dict["total"] + l1_term
            else:
                loss_dict = dict(loss_dict)
                loss_dict["l1"] = l1_term

            if sums is None:
                sums = {key: 0.0 for key in loss_dict}

            if training:
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(getattr(config, "GRAD_CLIP_NORM", 5.0)),
                )
                optimizer.step()

            bs = csi.shape[0]
            count += bs

            for key in sums:
                sums[key] += float(loss_dict[key].item()) * bs

            if pose_logits is not None and not _is_v2_model():
                predicted_class = pose_logits.argmax(dim=1)
                class_correct += int((predicted_class == pose_class).sum().item())
                class_count += bs
            elif _is_v2_model():
                occ = ~batch["is_empty"].to(device)
                if occ.any():
                    predicted_class = pose_logits.argmax(dim=1)
                    class_correct += int(
                        (predicted_class[occ] == pose_class[occ]).sum().item()
                    )
                    class_count += int(occ.sum().item())

    count = max(count, 1)

    if sums is None:
        sums = {}

    result = {key: value / count for key, value in sums.items()}
    result["classification_accuracy"] = (
        class_correct / class_count
        if class_count > 0
        else None
    )
    return result


def main():
    config.ensure_directories()
    set_seed(config.RANDOM_SEED)

    if not config.DATASET_FILE.exists():
        raise FileNotFoundError(
            "Dataset missing. Reuse/copy your existing csi_pose_dataset.npz "
            "or run python -m src.build_dataset."
        )

    # Same seed and same subject split as the 73.8% baseline.
    train_idx, val_idx, test_idx = create_split_indices(force=True)

    if config.NORMALIZE_CSI:
        compute_and_save_csi_stats(train_idx)

    train_ds = CSIPoseDataset(train_idx, augment=True)
    val_ds = CSIPoseDataset(val_idx)

    if getattr(config, "USE_SESSION_BALANCED_SAMPLER", False) and _is_v2_model():
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            sampler=SessionBalancedSampler(train_ds),
            num_workers=config.NUM_WORKERS,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = (
        ThreeHeadV2Loss()
        if _is_v2_model()
        else ConfidenceWeightedMultiTaskLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    if getattr(config, "LR_SCHEDULER", "") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.MIN_LEARNING_RATE,
        )
        plateau_scheduler = None
    else:
        scheduler = None
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(config, "LR_SCHEDULER_FACTOR", 0.5),
            patience=getattr(config, "LR_SCHEDULER_PATIENCE", 5),
            min_lr=config.MIN_LEARNING_RATE,
        )

    parameters = count_trainable_parameters(model)

    print("\nTRAINING - ACCURACY IMPROVEMENT V1")
    print("==================================")
    print(f"model: {config.MODEL_NAME}")
    print(f"device: {device}")
    print(f"parameters: {parameters:,}")
    print(f"split mode: {config.SPLIT_MODE}")
    print(
        f"window: {config.TIME_WINDOW_DURATION_S}s x "
        f"{config.TIME_WINDOW_SNAPSHOTS} | "
        f"CSI channels={getattr(config, 'CSI_INPUT_CHANNELS', 3)} | "
        f"median_k={getattr(config, 'CSI_MEDIAN_FILTER_KERNEL', 0)}"
    )
    print(
        "loss: "
        f"{config.LAMBDA_COORD}*coordinate + "
        f"{config.LAMBDA_RELATIVE}*relative + "
        f"{config.LAMBDA_CLASS}*classification + "
        f"L2(weight_decay={config.WEIGHT_DECAY}) + "
        f"L1(weight={config.L1_WEIGHT})"
    )
    print(
        "experiment flags: "
        f"representation={config.POSE_REPRESENTATION} | "
        f"decoder={config.SAM_DECODER} | "
        f"bone={config.LAMBDA_BONE} angle={config.LAMBDA_ANGLE} | "
        f"augment={config.AUGMENT_CSI} | "
        f"adversarial={config.ADVERSARIAL_SUBJECT} "
        f"(lambda={config.LAMBDA_SUBJECT}, grl={config.GRL_LAMBDA})"
    )
    print(
        f"early stopping patience={config.EARLY_STOPPING_PATIENCE}, "
        f"scheduler={getattr(config, 'LR_SCHEDULER', 'plateau')}"
    )

    history = []
    checkpoint_metric = getattr(
        config,
        "CHECKPOINT_SELECTION",
        "val_total_loss",
    )
    higher_is_better = checkpoint_metric == "val_classification_accuracy"
    best_score = float("-inf") if higher_is_better else float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    def _metric_value(result):
        if checkpoint_metric == "val_classification_accuracy":
            return float(result["classification_accuracy"])
        if checkpoint_metric == "val_total_loss":
            return float(result["total"])
        raise ValueError(
            f"Unknown CHECKPOINT_SELECTION={checkpoint_metric!r}"
        )

    print(
        f"checkpoint selection: {checkpoint_metric} "
        f"({'maximize' if higher_is_better else 'minimize'})"
    )

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_result = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        if scheduler is not None:
            scheduler.step()
        elif plateau_scheduler is not None:
            plateau_scheduler.step(val_result["total"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_result,
            "val": val_result,
        })

        train_acc = train_result["classification_accuracy"]
        val_acc = val_result["classification_accuracy"]

        if _is_v2_model():
            print(
                f"[{epoch:03d}/{config.NUM_EPOCHS}] "
                f"lr={current_lr:.2e} | "
                f"train={train_result['total']:.6f} "
                f"(presence={train_result.get('presence', 0):.4f}, "
                f"posture={train_result.get('posture', 0):.4f}, "
                f"pose={train_result.get('pose', 0):.4f}, "
                f"cls_acc={train_acc}) | "
                f"val={val_result['total']:.6f} "
                f"(cls_acc={val_acc})"
            )
        else:
            print(
                f"[{epoch:03d}/{config.NUM_EPOCHS}] "
                f"lr={current_lr:.2e} | "
                f"train={train_result['total']:.6f} "
                f"(xy={train_result.get('coordinate', 0):.6f}, "
                f"rel={train_result.get('relative', 0):.6f}, "
                f"cls={train_result.get('classification', 0):.6f}, "
                f"l1={train_result.get('l1', 0.0):.6f}, "
                f"cls_acc={train_acc:.3f}) | "
                f"val={val_result['total']:.6f} "
                f"(xy={val_result.get('coordinate', 0):.6f}, "
                f"rel={val_result.get('relative', 0):.6f}, "
                f"cls={val_result.get('classification', 0):.6f}, "
                f"cls_acc={val_acc:.3f})"
            )

        checkpoint = {
            "epoch": epoch,
            "model_name": config.MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_result,
            "parameters": parameters,
            "config_summary": {
                "lambda_coord": config.LAMBDA_COORD,
                "lambda_relative": config.LAMBDA_RELATIVE,
                "lambda_class": config.LAMBDA_CLASS,
                "weight_decay": config.WEIGHT_DECAY,
                "l1_weight": config.L1_WEIGHT,
                "dropout": config.CNN_DROPOUT,
                "bursts_per_window": config.BURSTS_PER_WINDOW,
            },
        }

        torch.save(checkpoint, config.LAST_MODEL_FILE)

        improved = (
            _metric_value(val_result)
            > best_score + config.EARLY_STOPPING_MIN_DELTA
            if higher_is_better
            else _metric_value(val_result)
            < best_score - config.EARLY_STOPPING_MIN_DELTA
        )

        if improved:
            best_score = _metric_value(val_result)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, config.BEST_MODEL_FILE)
            print(
                f"[train] BEST epoch {epoch} "
                f"({checkpoint_metric}={best_score:.6f}) "
                f"-> {config.BEST_MODEL_FILE}"
            )
        else:
            epochs_without_improvement += 1

        # Save history every epoch, so an interruption does not lose it.
        with config.TRAINING_HISTORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(
                "[train] early stopping: "
                f"no validation improvement for "
                f"{config.EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    print("\nTraining complete")
    print(f"best epoch: {best_epoch}")
    print(f"best {checkpoint_metric}: {best_score:.6f}")
    print(f"history: {config.TRAINING_HISTORY_FILE}")


if __name__ == "__main__":
    main()
