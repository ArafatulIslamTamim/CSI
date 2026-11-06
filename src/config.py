from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------- PATHS ----------------
RAW_VIDEO_DIR = ROOT / "data" / "raw" / "videos"
RAW_CSI_DIR = ROOT / "data" / "raw" / "csi"
FRAME_DIR = ROOT / "data" / "interim" / "frames"
POSE_LABEL_DIR = ROOT / "data" / "interim" / "pose_labels"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
QA_DIR = OUTPUT_DIR / "qa"

# ---------------- CSI ----------------
TX_ANT = 0
RX_INDICES = (0, 1, 2)
NUM_RX = 3
RAW_NUM_SUBCARRIERS = 64
ZERO_SUBCARRIER_INDICES = tuple(range(27, 38))
USABLE_SUBCARRIER_INDICES = tuple(
    i for i in range(RAW_NUM_SUBCARRIERS)
    if i not in ZERO_SUBCARRIER_INDICES
)
NUM_USABLE_SUBCARRIERS = len(USABLE_SUBCARRIER_INDICES)  # 53
VALID_BW = 0
VALID_SAMPLES = 64
VALID_MODE = 1
VALID_EXT = 0

CSI_FEATURE_MODE = "log_amp"
LOG_AMP_EPS = 1e-9
CSI_MEDIAN_FILTER_KERNEL = 3  # 0 = no filter; 3 = median-3 (LODO ablation)
CSI_INPUT_CHANNELS = 9  # 3 = amp only; 9 = amp + delta + rx_diff
EMPTY_ROOM_CAPTURE_KEY = "000_empty_room"
EMPTY_ROOM_EPS = 1e-6
EMPTY_ROOM_PROFILE_FILE = PROCESSED_DIR / "empty_room_profile.npz"

# LODO v2: five-day development set; today's capture is external blind test.
EXPERIMENT_TAG = "lodo_v2_9ch_med3"

# Three-head TCN: presence + posture + root-relative pose.
INCLUDE_EMPTY_ROOM_IN_DATASET = True

# Sessions excluded from dataset build (incomplete CSI / bad overlap).
EXCLUDED_SESSIONS = (
    "day5_004_standing_center_front",
)

NORMALIZE_CSI = True
CSI_NORM_EPS = 1e-8
CSI_LOCAL_NORMALIZATION = "none"  # v2: global per-channel/subcarrier stats only
# CSI_NORM_FILE is derived from EXPERIMENT_TAG in refresh_derived_paths().

# ---------------- ACTUAL BURST TIMING (WINDOW_MODE == "burst") ----------------
BURST_GAP_THRESHOLD_S = 0.020
# 5 CSI complete snapshots (3 Rx each) are grouped into one burst and
# aligned to one video frame.
SNAPSHOTS_PER_BURST = 5
MIN_SNAPSHOTS_PER_BURST = 3
MAX_SNAPSHOTS_PER_BURST = 8

# Temporal context: 5 consecutive bursts -> model input window.
BURSTS_PER_WINDOW = 5
MAX_BURST_WINDOW_SPAN_S = 1.0
# Half of a 5-FPS frame period (0.2 s) so bursts can match nearest frames.
MAX_BURST_TO_FRAME_SYNC_ERROR_S = 0.100
# Extract 5 image frames per second of video.
TARGET_VIDEO_FPS = 5.0

# ---------------- WINDOWING ----------------
# "burst": legacy K bursts x S snapshots (default CNN+GRU input).
# "time_causal": host_ts window resampled to TIME_WINDOW_SNAPSHOTS (TCN).
WINDOW_MODE = "time_causal"
TIME_WINDOW_DURATION_S = 0.6
TIME_WINDOW_SNAPSHOTS = 32
TIME_WINDOW_MIN_SNAPSHOTS = 3
TIME_WINDOW_STRIDE_S = 0.3  # 50% overlap for 600 ms windows
TIME_WINDOW_LINEAR_INTERP = True
TIME_WINDOW_MIN_RECORDS_PER_RX = 18
TIME_WINDOW_MAX_GAP_S = 0.10
TIME_WINDOW_MAX_SYNC_ERROR_S = 0.100
LIVE_WINDOW_EMIT_INTERVAL_S = 0.2

# ---------------- YOLO-POSE TEACHER ----------------
NUM_KEYPOINTS = 17
YOLO_POSE_MODEL = MODEL_DIR / "yolo26n-pose.pt"
YOLO_POSE_FALLBACK_NAME = "yolo26n-pose.pt"
YOLO_DEVICE = "cpu"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.25
MIN_MEAN_KEYPOINT_CONFIDENCE = 0.0
YOLO_ANNOTATE_ONLY_SYNC_FRAMES = True
SAVE_YOLO_VISUALIZATIONS = True

# ---------------- SAM ----------------
SAM_CHANNELS = 2

# ---------------- MODEL ----------------
# Improvement V1:
# CNN -> GRU -> shared representation
#                |-> SAM regression head
#                |-> Sitting/Standing auxiliary classification head
MODEL_NAME = "tcn_three_head_v2"
VALID_MODELS = {
    "cnn_fc_sam",
    "cnn_gru_sam",
    "cnn_gru_multitask_sam",
    "tcn_multitask_sam",
    "tcn_three_head_v2",
    "transformer_sam",
}
if MODEL_NAME not in VALID_MODELS:
    raise ValueError(f"MODEL_NAME must be one of {sorted(VALID_MODELS)}")

CNN_CHANNELS = (32, 64, 128)
CNN_FEATURE_DIM = 128

# Slightly stronger regularization than Pipeline 2 baseline (0.20).
CNN_DROPOUT = 0.30

GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS = 1
GRU_BIDIRECTIONAL = False
GRU_DROPOUT = 0.0

TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_LAYERS = 4
TRANSFORMER_FF_DIM = 256
TRANSFORMER_DROPOUT = 0.10
FC_HIDDEN = 256

# Posture head (occupied windows only): sitting=0, standing=1.
POSE_CLASSES = ("sitting", "standing")
POSE_TO_CLASS = {name: i for i, name in enumerate(POSE_CLASSES)}
NUM_POSE_CLASSES = len(POSE_CLASSES)

EMPTY_POSE_NAME = "empty"
EMPTY_CLASS_INDEX = -1  # presence head handles empty vs occupied
EMPTY_SUBJECT_ID = "000"

# Label QC (YOLO teacher).
MIN_VISIBLE_KEYPOINTS = 8
MIN_KEYPOINT_CONFIDENCE = 0.25
YOLO_LABEL_MEDIAN_KERNEL = 3

# Empty-room: mix day1–day5 into TRAIN (avoid single-day memorization).
# Hold out empties for VAL/TEST, including one day5 session as new-domain test.
EMPTY_TRAIN_SESSIONS = (
    "day1_000_empty_room",
    "day1_000_empty_room_2",
    "day1_000_empty_room_3",
    "day1_000_empty_room_4",
    "day1_000_empty_room_5",
    "day2_000_empty_room",
    "day2_000_empty_room_2",
    "day3_000_empty_room",
    "day3_000_empty_room_2",
    "day4_000_empty_room",
    "day4_000_empty_room_2",
    "day5_000_empty_room",
    "day5_000_empty_room_2",
    "day5_000_empty_room_3",
)
EMPTY_VAL_SESSIONS = (
    "day2_000_empty_room_3",
    "day5_000_empty_room_4",
)
EMPTY_TEST_SESSIONS = (
    "day3_000_empty_room_3",
    "day4_000_empty_room_3",
    "day5_000_empty_room_5",
)
# Disabled: explicit lists above (do not auto-split by day).
EMPTY_DAY_SPLIT = {}

# TCN temporal encoder (WINDOW_MODE == "time_causal").
TCN_HIDDEN_SIZE = 128
TCN_NUM_LAYERS = 4
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = (1, 2, 4, 8)

# ---------------- LOSS ----------------
# Baseline used relative weight 0.20.
# Sitting showed large coordinate error, so V1 gives absolute coordinates more priority.
LAMBDA_COORD = 1.0
LAMBDA_RELATIVE = 0.10
# Stronger classification for empty-room gating (was 0.10 on day3_id007).
LAMBDA_CLASS = 1.0
# Per-class CE weights: (sitting, standing).
CLASS_LOSS_WEIGHTS = (1.0, 1.0)
MIN_CONFIDENCE_WEIGHT = 0.05

# ---------------- POSE REPRESENTATION (P1) ----------------
# "absolute": SAM diagonal carries absolute normalized coordinates (baseline).
# "root_relative": SAM diagonal carries (xy - root) / body_scale and a
# separate root head predicts (root_x, root_y, log_scale), so person
# localization errors no longer contaminate pose geometry supervision.
POSE_REPRESENTATION = "root_relative_v2"
ROOT_JOINT_INDICES = (11, 12)      # hips -> root = mid-hip (bbox root used in v2)
SHOULDER_JOINT_INDICES = (5, 6)
MIN_BODY_SCALE = 0.05
# v2 combined loss weights (ThreeHeadV2Loss).
LAMBDA_POSE = 1.0
LAMBDA_ROOT = 0.20
LAMBDA_SCALE = 0.10
LAMBDA_BONE = 0.05
LAMBDA_POSTURE = 0.20
LAMBDA_PRESENCE = 0.30
LAMBDA_RELPOSE = 0.5               # legacy SAM path

# ---------------- GEOMETRY LOSS (P2) ----------------
# Per-bone / per-angle terms instead of one pooled similarity score:
# each term keeps a clean gradient to the joints it constrains.
LAMBDA_BONE = 0.0    # L1 on the 16 COCO bone lengths (0 disables)
LAMBDA_ANGLE = 0.0   # MSE on cos(joint angle) at elbows/knees/hips (0 disables)

# ---------------- CSI AUGMENTATION (P3) ----------------
# Applied on the training split only, after normalization.
AUGMENT_CSI = True
AUG_NOISE_STD = 0.035
AUG_SUBCARRIER_DROP_PROB = 0.5
AUG_SUBCARRIER_DROP_COUNT = (3, 5)  # mask 3-5 valid bins when triggered
AUG_SUBCARRIER_DROP_FRAC = 0.10
AUG_RX_GAIN_STD = 0.175          # ~±1.5 dB independent Rx
AUG_COMMON_GAIN_DB = 1.0         # uniform ±1 dB log-amplitude
AUG_TIME_MASK_PROB = 0.3
AUG_TIME_MASK_POINTS = (1, 2)
AUG_TIMESTAMP_JITTER_MS = 20.0
AUG_PACKET_DROP_FRAC = 0.05
AUG_RX_DROPOUT_PROB = 0.05

# ---------------- SUBJECT-ADVERSARIAL TRAINING (P4) ----------------
# A subject classifier behind a gradient-reversal layer pushes the shared
# CSI representation to be subject-invariant (empty-room samples excluded).
ADVERSARIAL_SUBJECT = False
LAMBDA_SUBJECT = 0.10
GRL_LAMBDA = 1.0
SUBJECT_IDS = ("001", "002", "003", "004", "005", "006", "007")

# ---------------- SAM DECODER (P5) ----------------
# "fc": baseline fully-connected SAM head.
# "gcn": graph-convolutional decoder over the COCO skeleton adjacency that
# predicts per-joint coordinates; the SAM tensor is built from them.
SAM_DECODER = "fc"
GCN_NODE_DIM = 32
GCN_LAYERS = 2

# ---------------- TRAIN / SPLIT ----------------
RANDOM_SEED = 42
# "loso_day": leave-one-day-out (LODO_HOLDOUT_DAY); "final_all_days": train all 5.
SPLIT_MODE = "loso_day"
LODO_HOLDOUT_DAY = "day1"
ALL_DAYS = ("day1", "day2", "day3", "day4", "day5")

# Legacy fixed day split (stress test).
FIXED_TRAIN_DAYS = ("day1", "day2", "day5")
FIXED_VAL_DAYS = ("day3",)
FIXED_TEST_DAYS = ("day4",)

# Used only when SPLIT_MODE == "fixed_subject".
FIXED_TEST_SUBJECTS = ("005",)
FIXED_VAL_SUBJECTS = ("006",)

# Kept only as fallback if SPLIT_MODE is changed back to "subject".
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 5
GRAD_CLIP_NORM = 1.0

WEIGHT_DECAY = 1e-4
CSI_NORM_CLIP = 6.0

USE_SESSION_BALANCED_SAMPLER = True
BATCH_EMPTY_FRAC = 0.20
# L1 penalty on all trainable weights, added to the training loss.
# 0 disables. Keep small so pose/class heads are not over-shrunk.
L1_WEIGHT = 1e-5
NUM_WORKERS = 0

# Early stopping and LR scheduler target the unstable validation behavior seen
# after the best baseline epoch.
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-6
LR_SCHEDULER = "cosine"
MIN_LEARNING_RATE = 1e-6

# LODO selection score: 0.5*worst_day_pck + 0.3*mean_pck + 0.2*presence_f1
CHECKPOINT_SELECTION = "lodo_score"
FINAL_TRAIN_EPOCHS = None  # set from median LODO best epochs before final run

# ---------------- EVALUATION ----------------
PCK_ALPHA = 0.20
PIXEL_PCK_THRESHOLDS = (5, 10, 20, 30, 40, 50)

# ---------------- LIVE INFERENCE ----------------
# Skeleton output is suppressed when P(empty) exceeds this threshold.
LIVE_EMPTY_PROB_THRESHOLD = 0.40
LIVE_OCCUPIED_PROB_THRESHOLD = 0.75
LIVE_PRESENCE_ENTER_COUNT = 3
LIVE_PRESENCE_EXIT_COUNT = 5

LIVE_CLASS_VOTE_WINDOW = 3
LIVE_KEYPOINT_EMA_ALPHA = 0.3

# A burst still being accumulated is force-closed if no new snapshot
# arrives within this wall-clock timeout.
LIVE_BURST_IDLE_TIMEOUT_S = 0.10

LIVE_DISPLAY_WIDTH = 640
LIVE_DISPLAY_HEIGHT = 480

# ---------------- OUTPUTS ----------------
DATASET_FILE = PROCESSED_DIR / "csi_pose_dataset.npz"


def refresh_derived_paths():
    """Recompute every path that depends on EXPERIMENT_TAG / MODEL_NAME."""
    global CSI_NORM_FILE, SPLIT_FILE, BEST_MODEL_FILE, LAST_MODEL_FILE
    global TRAINING_HISTORY_FILE, METRICS_FILE, TEST_PREDICTIONS_FILE
    global PER_POSE_METRICS_FILE

    CSI_NORM_FILE = (
        OUTPUT_DIR / f"csi_normalization_stats_{EXPERIMENT_TAG}.npz"
    )
    SPLIT_FILE = OUTPUT_DIR / f"split_indices_{EXPERIMENT_TAG}.npz"
    BEST_MODEL_FILE = CHECKPOINT_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_best.pt"
    LAST_MODEL_FILE = CHECKPOINT_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_last.pt"
    TRAINING_HISTORY_FILE = (
        OUTPUT_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_training_history.json"
    )
    METRICS_FILE = EVALUATION_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_metrics.json"
    TEST_PREDICTIONS_FILE = (
        EVALUATION_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_test_predictions.npz"
    )
    PER_POSE_METRICS_FILE = (
        EVALUATION_DIR / f"{MODEL_NAME}_{EXPERIMENT_TAG}_metrics_by_pose.json"
    )


refresh_derived_paths()


# ---------------- EXPERIMENT OVERRIDES ----------------
# scripts/run_loso_experiment.py points the CSI_CONFIG_OVERRIDES environment
# variable at a JSON file of {config_name: value} pairs, so leave-one-subject-
# out folds and experiment variants never rewrite this module on disk.
def _apply_environment_overrides():
    import json
    import os

    overrides_path = os.environ.get("CSI_CONFIG_OVERRIDES", "")
    if not overrides_path:
        return

    with open(overrides_path, "r", encoding="utf-8") as f:
        overrides = json.load(f)

    for key, value in overrides.items():
        if key not in globals():
            raise KeyError(f"Unknown config override: {key}")
        if isinstance(globals()[key], tuple) and isinstance(value, list):
            value = tuple(value)
        globals()[key] = value

    if MODEL_NAME not in VALID_MODELS:
        raise ValueError(f"MODEL_NAME must be one of {sorted(VALID_MODELS)}")

    refresh_derived_paths()


_apply_environment_overrides()


def ensure_directories():
    for path in [
        RAW_VIDEO_DIR, RAW_CSI_DIR, FRAME_DIR, POSE_LABEL_DIR,
        PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
        EVALUATION_DIR, VISUALIZATION_DIR, QA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)