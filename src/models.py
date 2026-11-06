import torch
from torch import nn

from . import config
from .geometry import build_sam_from_keypoints, skeleton_adjacency


OUTPUT_DIM = (
    config.SAM_CHANNELS
    * config.NUM_KEYPOINTS
    * config.NUM_KEYPOINTS
)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Identity forward; multiplies the gradient by -lambda on backward."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.lambda_)


class _GCNLayer(nn.Module):
    """One graph-convolution layer over a fixed normalized adjacency."""

    def __init__(self, dim):
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_neighbor = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adjacency):
        h = self.w_self(x) + torch.einsum(
            "ij,bjd->bid",
            adjacency,
            self.w_neighbor(x),
        )
        return x + torch.relu(self.norm(h))


class GCNDecoder(nn.Module):
    """
    Graph-convolutional SAM decoder (P5).

    The temporal feature is projected to one embedding per COCO joint;
    message passing over the skeleton adjacency refines them; a shared
    linear layer emits per-joint (x, y). The SAM tensor is rebuilt from
    the coordinates so downstream losses/metrics are unchanged.
    """

    def __init__(self, in_dim):
        super().__init__()

        k = config.NUM_KEYPOINTS
        d = config.GCN_NODE_DIM

        self.node_init = nn.Linear(in_dim, k * d)
        self.node_embedding = nn.Parameter(torch.zeros(k, d))
        nn.init.normal_(self.node_embedding, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            _GCNLayer(d) for _ in range(config.GCN_LAYERS)
        )

        self.out = nn.Linear(d, 2)

        self.register_buffer(
            "adjacency",
            skeleton_adjacency(k),
        )

    def forward(self, feature):
        b = feature.shape[0]
        x = self.node_init(feature).view(
            b,
            config.NUM_KEYPOINTS,
            config.GCN_NODE_DIM,
        )
        x = x + self.node_embedding

        for layer in self.layers:
            x = layer(x, self.adjacency)

        coordinates = self.out(x)
        return build_sam_from_keypoints(coordinates)


class BurstCNNEncoder(nn.Module):
    """Encode one CSI burst [B,5,3,53] into one feature vector."""

    def __init__(self):
        super().__init__()

        c1, c2, c3 = config.CNN_CHANNELS

        self.features = nn.Sequential(
            nn.Conv2d(
                config.NUM_RX,
                c1,
                kernel_size=(3, 5),
                padding=(1, 2),
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                c1,
                c2,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(
                c2,
                c3,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 4)),
        )

        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 2 * 4, config.CNN_FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
        )

    def forward(self, x):
        # [B,5,3,53] -> [B,3,5,53]
        x = x.permute(0, 2, 1, 3).contiguous()
        return self.project(self.features(x))


class CNNFCSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BurstCNNEncoder()
        self.head = nn.Sequential(
            nn.Linear(config.CNN_FEATURE_DIM, config.FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(config.FC_HIDDEN, OUTPUT_DIM),
        )

    def forward(self, x):
        feature = self.encoder(x[:, -1])
        y = self.head(feature)
        return y.view(
            -1,
            config.SAM_CHANNELS,
            config.NUM_KEYPOINTS,
            config.NUM_KEYPOINTS,
        )


class _CNNGRUBackbone(nn.Module):
    """Shared CNN + GRU backbone used by both GRU variants."""

    def __init__(self):
        super().__init__()
        self.encoder = BurstCNNEncoder()

        self.gru = nn.GRU(
            input_size=config.CNN_FEATURE_DIM,
            hidden_size=config.GRU_HIDDEN_SIZE,
            num_layers=config.GRU_NUM_LAYERS,
            batch_first=True,
            dropout=(
                config.GRU_DROPOUT
                if config.GRU_NUM_LAYERS > 1
                else 0.0
            ),
            bidirectional=config.GRU_BIDIRECTIONAL,
        )

        self.output_dim = config.GRU_HIDDEN_SIZE * (
            2 if config.GRU_BIDIRECTIONAL else 1
        )

    def forward(self, x):
        # [B,K,5,3,53]
        b, k, s, rx, sub = x.shape

        flat = x.reshape(b * k, s, rx, sub)
        burst_features = self.encoder(flat)
        sequence = burst_features.reshape(b, k, -1)

        output, _ = self.gru(sequence)
        return output[:, -1, :]


class CNNGRUSAM(nn.Module):
    """Original Pipeline-2 single-task CNN-GRU baseline."""

    def __init__(self):
        super().__init__()
        self.backbone = _CNNGRUBackbone()
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, config.FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(config.FC_HIDDEN, OUTPUT_DIM),
        )

    def forward(self, x):
        temporal = self.backbone(x)
        y = self.head(temporal)
        return y.view(
            -1,
            config.SAM_CHANNELS,
            config.NUM_KEYPOINTS,
            config.NUM_KEYPOINTS,
        )


class CNNGRUMultiTaskSAM(nn.Module):
    """
    Accuracy Improvement V1.

    Shared CSI representation:
        CNN per burst -> GRU

    Two training objectives:
        1) SAM/keypoint regression
        2) sitting/standing classification

    The classification head is auxiliary: CSI-only pose regression remains
    the final task/output used for skeleton prediction.
    """

    def __init__(self):
        super().__init__()
        self.backbone = _CNNGRUBackbone()
        d = self.backbone.output_dim

        if config.SAM_DECODER == "gcn":
            self.sam_head = GCNDecoder(d)
        elif config.SAM_DECODER == "fc":
            self.sam_head = nn.Sequential(
                nn.Linear(d, config.FC_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Dropout(config.CNN_DROPOUT),
                nn.Linear(config.FC_HIDDEN, OUTPUT_DIM),
            )
        else:
            raise ValueError(
                f"Unknown SAM_DECODER={config.SAM_DECODER!r}. Use 'fc' or 'gcn'."
            )

        self.class_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(64, config.NUM_POSE_CLASSES),
        )

        # P1: predict (root_x, root_y, log_scale) separately from the
        # root-relative pose so localization does not contaminate geometry.
        if config.POSE_REPRESENTATION == "root_relative":
            self.root_head = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.CNN_DROPOUT),
                nn.Linear(64, 3),
            )
        else:
            self.root_head = None

        # P4: subject discriminator behind a gradient-reversal layer.
        if config.ADVERSARIAL_SUBJECT:
            self.gradient_reversal = GradientReversal(config.GRL_LAMBDA)
            self.subject_head = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.CNN_DROPOUT),
                nn.Linear(64, len(config.SUBJECT_IDS)),
            )
        else:
            self.gradient_reversal = None
            self.subject_head = None

    def forward(self, x):
        temporal = self.backbone(x)

        sam = self.sam_head(temporal).view(
            -1,
            config.SAM_CHANNELS,
            config.NUM_KEYPOINTS,
            config.NUM_KEYPOINTS,
        )

        pose_logits = self.class_head(temporal)

        output = {
            "sam": sam,
            "pose_logits": pose_logits,
        }

        if self.root_head is not None:
            output["root"] = self.root_head(temporal)

        if self.subject_head is not None:
            output["subject_logits"] = self.subject_head(
                self.gradient_reversal(temporal)
            )

        return output


class SnapshotCNNEncoder(nn.Module):
    """Encode one CSI snapshot [B, Rx, Sub] -> feature vector."""

    def __init__(self):
        super().__init__()
        self._burst = BurstCNNEncoder()

    def forward(self, x):
        # [B, 3, 53] -> treat as one snapshot in a burst
        return self._burst(x.unsqueeze(1))


class _CausalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        out = self.conv(
            nn.functional.pad(x, (self.pad, 0))
        )
        return x + torch.relu(self.norm(out))


class TCNTemporalEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = config.TCN_HIDDEN_SIZE
        self.input_proj = nn.Linear(input_dim, hidden)
        dilations = tuple(config.TCN_DILATIONS)
        blocks = []
        for dilation in dilations[:config.TCN_NUM_LAYERS]:
            blocks.append(
                _CausalConvBlock(
                    hidden,
                    config.TCN_KERNEL_SIZE,
                    dilation,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = hidden

    def forward(self, sequence):
        # [B, T, D] -> [B, D]
        x = self.input_proj(sequence)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return x[:, :, -1]


class TCNMultiTaskSAM(nn.Module):
    """500ms causal windows: per-snapshot CNN + residual TCN + multitask heads."""

    def __init__(self):
        super().__init__()
        self.snapshot_encoder = SnapshotCNNEncoder()
        self.temporal = TCNTemporalEncoder(config.CNN_FEATURE_DIM)
        d = self.temporal.output_dim

        if config.SAM_DECODER == "gcn":
            self.sam_head = GCNDecoder(d)
        elif config.SAM_DECODER == "fc":
            self.sam_head = nn.Sequential(
                nn.Linear(d, config.FC_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Dropout(config.CNN_DROPOUT),
                nn.Linear(config.FC_HIDDEN, OUTPUT_DIM),
            )
        else:
            raise ValueError(
                f"Unknown SAM_DECODER={config.SAM_DECODER!r}. Use 'fc' or 'gcn'."
            )

        self.class_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(64, config.NUM_POSE_CLASSES),
        )

        if config.POSE_REPRESENTATION == "root_relative":
            self.root_head = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.CNN_DROPOUT),
                nn.Linear(64, 3),
            )
        else:
            self.root_head = None

        if config.ADVERSARIAL_SUBJECT:
            self.gradient_reversal = GradientReversal(config.GRL_LAMBDA)
            self.subject_head = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(config.CNN_DROPOUT),
                nn.Linear(64, len(config.SUBJECT_IDS)),
            )
        else:
            self.gradient_reversal = None
            self.subject_head = None

    def forward(self, x):
        # [B, T, Rx, Sub]
        b, t, rx, sub = x.shape
        flat = x.reshape(b * t, rx, sub)
        snapshot_features = self.snapshot_encoder(flat)
        sequence = snapshot_features.reshape(b, t, -1)
        temporal = self.temporal(sequence)

        if config.SAM_DECODER == "gcn":
            sam = self.sam_head(temporal)
        else:
            sam = self.sam_head(temporal).view(
                -1,
                config.SAM_CHANNELS,
                config.NUM_KEYPOINTS,
                config.NUM_KEYPOINTS,
            )

        pose_logits = self.class_head(temporal)
        output = {
            "sam": sam,
            "pose_logits": pose_logits,
        }

        if self.root_head is not None:
            output["root"] = self.root_head(temporal)

        if self.subject_head is not None:
            output["subject_logits"] = self.subject_head(
                self.gradient_reversal(temporal)
            )

        return output


class _CausalConvBlockV2(nn.Module):
    """Two-layer causal residual block with dropout (v2 TCN)."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.25):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.drop = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x):
        residual = x
        out = self.conv1(nn.functional.pad(x, (self.pad, 0)))
        out = self.drop(torch.nn.functional.gelu(self.norm1(out)))
        out = self.conv2(nn.functional.pad(out, (self.pad, 0)))
        out = self.drop(torch.nn.functional.gelu(self.norm2(out)))
        return residual + out


class SpatialSubcarrierEncoderV2(nn.Module):
    """Conv1D over subcarriers: [B, C_in, S] -> [B, 96]."""

    def __init__(self, in_channels=None):
        super().__init__()
        c_in = int(
            in_channels
            if in_channels is not None
            else getattr(config, "CSI_INPUT_CHANNELS", 9)
        )
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 32, kernel_size=5, padding=2),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = 96

    def forward(self, x):
        # x: [B, C, S]
        h = self.net(x)
        return h.squeeze(-1)


class TCNThreeHeadV2(nn.Module):
    """
    v2 pipeline: 600 ms causal windows, 9-channel CSI, three heads:
    presence (1), posture (2), root-relative pose (3 + 34 = 37).
    """

    def __init__(self):
        super().__init__()
        in_ch = int(getattr(config, "CSI_INPUT_CHANNELS", 9))
        self.spatial = SpatialSubcarrierEncoderV2(in_ch)
        hidden = config.TCN_HIDDEN_SIZE
        self.temporal_in = nn.Linear(self.spatial.out_dim, hidden)
        dilations = tuple(config.TCN_DILATIONS)[: config.TCN_NUM_LAYERS]
        self.temporal_blocks = nn.ModuleList(
            [
                _CausalConvBlockV2(
                    hidden,
                    config.TCN_KERNEL_SIZE,
                    d,
                    dropout=0.25,
                )
                for d in dilations
            ]
        )
        d = hidden
        self.presence_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )
        self.posture_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, config.NUM_POSE_CLASSES),
        )
        self.root_scale_head = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 3),
        )
        self.rel_pose_head = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, config.NUM_KEYPOINTS * 2),
        )

    def forward(self, x):
        # x: [B, T, C, S]
        b, t, c, s = x.shape
        flat = x.reshape(b * t, c, s)
        snap_feat = self.spatial(flat).reshape(b, t, -1)
        h = self.temporal_in(snap_feat).transpose(1, 2)
        for block in self.temporal_blocks:
            h = block(h)
        feat = h[:, :, -1]

        rel = self.rel_pose_head(feat).view(b, config.NUM_KEYPOINTS, 2)
        return {
            "presence_logit": self.presence_head(feat),
            "posture_logits": self.posture_head(feat),
            "root_scale": self.root_scale_head(feat),
            "rel_pose": rel,
        }


class TransformerSAM(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = config.NUM_RX * config.NUM_USABLE_SUBCARRIERS

        self.embedding = nn.Linear(
            input_dim,
            config.TRANSFORMER_D_MODEL,
        )

        max_tokens = (
            config.BURSTS_PER_WINDOW
            * config.SNAPSHOTS_PER_BURST
        )

        self.positional = nn.Parameter(
            torch.zeros(
                1,
                max_tokens,
                config.TRANSFORMER_D_MODEL,
            )
        )
        nn.init.normal_(self.positional, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            dim_feedforward=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.TRANSFORMER_LAYERS,
        )
        self.norm = nn.LayerNorm(config.TRANSFORMER_D_MODEL)

        self.head = nn.Sequential(
            nn.Linear(config.TRANSFORMER_D_MODEL, config.FC_HIDDEN),
            nn.GELU(),
            nn.Dropout(config.TRANSFORMER_DROPOUT),
            nn.Linear(config.FC_HIDDEN, OUTPUT_DIM),
        )

    def forward(self, x):
        b, k, s, rx, sub = x.shape
        tokens = x.reshape(b, k * s, rx * sub)
        h = self.embedding(tokens)
        h = h + self.positional[:, :h.shape[1], :]
        h = self.encoder(h)
        pooled = self.norm(h.mean(dim=1))
        y = self.head(pooled)

        return y.view(
            -1,
            config.SAM_CHANNELS,
            config.NUM_KEYPOINTS,
            config.NUM_KEYPOINTS,
        )


def unpack_model_output(output):
    """Return (predicted_sam, pose_logits_or_None) for all model variants."""
    if isinstance(output, dict):
        if "sam" in output:
            return output["sam"], output.get("pose_logits")
        if "posture_logits" in output:
            return output, output.get("posture_logits")
    return output, None


def reconstruct_keypoints_from_v2(root_scale, rel_pose):
    """root_scale [B,3]=(rx,ry,s); rel_pose [B,K,2]."""
    root_xy = root_scale[:, :2].unsqueeze(1)
    scale = root_scale[:, 2:3].unsqueeze(1).clamp_min(config.MIN_BODY_SCALE)
    return root_xy + scale * rel_pose


def build_model():
    root_relative_models = {
        "cnn_gru_multitask_sam",
        "tcn_multitask_sam",
        "tcn_three_head_v2",
    }
    if (
        config.POSE_REPRESENTATION in ("root_relative", "root_relative_v2")
        and config.MODEL_NAME not in root_relative_models
    ):
        raise ValueError(
            "POSE_REPRESENTATION='root_relative' requires "
            f"MODEL_NAME in {sorted(root_relative_models)}."
        )

    if config.MODEL_NAME == "cnn_fc_sam":
        return CNNFCSAM()
    if config.MODEL_NAME == "cnn_gru_sam":
        return CNNGRUSAM()
    if config.MODEL_NAME == "cnn_gru_multitask_sam":
        return CNNGRUMultiTaskSAM()
    if config.MODEL_NAME == "tcn_multitask_sam":
        return TCNMultiTaskSAM()
    if config.MODEL_NAME == "tcn_three_head_v2":
        return TCNThreeHeadV2()
    if config.MODEL_NAME == "transformer_sam":
        return TransformerSAM()
    raise ValueError(f"Unknown MODEL_NAME={config.MODEL_NAME}")


def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
