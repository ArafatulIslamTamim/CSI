# LODO v2 pipeline protocol

This document mirrors the frozen development protocol for Days 1–5 and
today's external blind test.

## Principles

1. **Days 1–5** = complete development dataset.
2. **Today's capture** = external blind test (one-shot evaluation).
3. All filters, thresholds, architecture, and epoch counts are chosen
   **before** looking at today's predictions.

## Pipeline stages

| Stage | Script | Purpose |
|-------|--------|---------|
| Clean + label QC | `src/build_dataset` | Drop bad session, malformed records, stride windows |
| LODO selection | `scripts/run_lodo_experiment.py` | Pick filter, channels, epochs |
| Final train | `--final` | All five days, fixed epoch count |
| Blind test | external eval | Today only, no retuning |

## Data cleaning (automatic)

- **Remove session:** `day5_004_standing_center_front` (`EXCLUDED_SESSIONS`)
- **Malformed records:** dropped in `csi_parser.is_valid_record()` (~168 total)
- **Zero subcarriers:** indices 27–37 removed (53 usable bins)
- **Do not** auto-remove statistical CSI outliers

## Model

`tcn_three_head_v2`:

- **Presence** head (empty vs occupied)
- **Posture** head (sitting vs standing, occupied only)
- **Pose** head (root + scale + 17×2 relative keypoints)

Input: `[B, 32, 9, 53]` with log-amplitude features.

## Splits

| Mode | Train | Val |
|------|-------|-----|
| `loso_day` | 4 days | 1 held-out day |
| `final_all_days` | all 5 days | none (fixed epochs) |

## Normalization

- LODO: stats from **4 training days** in each fold only.
- Final: stats from **all cleaned Days 1–5**.
- Clip normalized values to **±6**.
- **Never** use today's data for normalization.

## First experiment matrix

Keep everything identical except input channels:

1. `amp3_nomed` — 3 channels, no median filter
2. `amp9_med3` — 9 channels, median-3

Select using worst-day PCK + mean PCK + presence F1 (see spec §14).

## Related files

- `src/config.py` — defaults for v2
- `src/csi_features.py` — log amp, median, 9-channel stack
- `src/samplers.py` — session-balanced batches
- `outputs/qa/five_day_csi_diversity_analysis.json` — data audit
- `commands_lodo_v2.txt` — shell commands
