"""Load cnn_gru_multitask_sam checkpoint and run live inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config
from src.losses import reconstruct_absolute_keypoints
from src.models import build_model, unpack_model_output

from model import EMPTY_CLASS_NAME, POSE_CLASSES

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "best_model.pt"


class PoseEngine:
    def __init__(
        self,
        checkpoint: Path | str = DEFAULT_CHECKPOINT,
        device: Optional[str] = None,
    ):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint}\n"
                "Copy your trained .pt into pose_detection_live/best_model.pt "
                "or pass --checkpoint /path/to/model.pt"
            )

        if not config.CSI_NORM_FILE.exists():
            raise FileNotFoundError(
                f"CSI norm stats missing: {config.CSI_NORM_FILE}\n"
                "Build/train the dataset first so outputs/csi_normalization_stats_*.npz exists."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        try:
            ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.checkpoint, map_location=self.device)

        model_name = ckpt.get("model_name") or ckpt.get("model_type")
        if model_name and model_name != config.MODEL_NAME:
            # Allow GUI launch with CSI_CONFIG_OVERRIDES matching the checkpoint.
            import os
            if not os.environ.get("CSI_CONFIG_OVERRIDES"):
                raise ValueError(
                    f"Unsupported checkpoint model_name={model_name!r}. "
                    f"Expected {config.MODEL_NAME!r}. "
                    "Set CSI_CONFIG_OVERRIDES=pose_detection_live/tcn_gui_overrides.json "
                    "for TCN checkpoints."
                )
            config.MODEL_NAME = model_name

        state = ckpt.get("model_state_dict")
        if state is None:
            raise KeyError("Checkpoint missing model_state_dict")

        self.model = build_model().to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # Live preprocessor loads the same train-set z-score file; keep a
        # lightweight marker so the GUI knows norm is available.
        self.csi_norm = {
            "mode": "train_set_file",
            "path": str(config.CSI_NORM_FILE),
            "gain_norm_mode": None,
            "empty_mean": None,
            "use_log1p": False,
        }
        self.empty_calibrated = False
        self._checkpoint_empty_mean = None

        self.epoch = int(ckpt.get("epoch", -1))
        val = ckpt.get("val_loss")
        if isinstance(val, dict):
            self.val_loss = float(val.get("total", float("nan")))
            self.val_pose_acc = float(val.get("classification_accuracy", float("nan")))
        else:
            self.val_loss = float(val) if val is not None else float("nan")
            self.val_pose_acc = float("nan")
        self.val_pck = float(ckpt.get("val_pck", float("nan")))
        self.num_subjects = 0
        self.model_name = model_name or config.MODEL_NAME

    def calibrate_empty_mean(self, empty_mean: np.ndarray) -> dict:
        """Amplitude multitask model does not use an empty residual."""
        raise RuntimeError(
            "This model uses amplitude + train-set z-score "
            f"({config.CSI_NORM_FILE.name}). Empty is a classifier class — "
            "no live empty residual calibration is needed."
        )

    def reset_empty_mean(self) -> None:
        self.empty_calibrated = False

    def checkpoint_empty_mean_value(self) -> float | None:
        return None

    @torch.no_grad()
    def predict(self, sample: np.ndarray) -> dict:
        """
        Run inference on one normalized CSI window.

        Parameters
        ----------
        sample : ndarray
            Burst mode: (K, S, RX, Sub) = (5, 5, 3, 53)
            TCN time-causal: (T, RX, Sub) = (32, 3, 53)
        """
        x = torch.from_numpy(np.ascontiguousarray(sample)).float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        out = self.model(x)
        pred_sam, pose_logits = unpack_model_output(out)
        root = out.get("root") if isinstance(out, dict) else None
        keypoints = (
            reconstruct_absolute_keypoints(pred_sam, root)[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if pose_logits is None:
            probs = np.zeros(len(POSE_CLASSES), dtype=np.float32)
            pose_id = 0
        else:
            probs = torch.softmax(pose_logits[0], dim=-1).cpu().numpy().astype(np.float32)
            pose_id = int(pose_logits[0].argmax().item())

        pose_name = POSE_CLASSES[pose_id]
        if (
            config.EMPTY_CLASS_INDEX >= 0
            and (
                pose_name == EMPTY_CLASS_NAME
                or pose_id == config.EMPTY_CLASS_INDEX
            )
        ):
            keypoints = np.zeros_like(keypoints)

        return {
            "keypoints": keypoints,
            "pose_id": pose_id,
            "pose_name": pose_name,
            "pose_probs": probs,
        }

    def info(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "epoch": self.epoch,
            "val_loss": self.val_loss,
            "val_pck": self.val_pck,
            "val_pose_acc": self.val_pose_acc,
            "num_subjects": self.num_subjects,
            "has_csi_norm": True,
            "gain_norm_mode": "train_zscore",
            "empty_calibrated": self.empty_calibrated,
            "pose_classes": list(POSE_CLASSES),
            "model_name": self.model_name,
            "csi_norm_file": str(config.CSI_NORM_FILE),
        }
