# WiFi CSI Human Pose Estimation — YOLO-Pose Version

Camera ground truth is generated with pretrained Ultralytics YOLO-Pose using the COCO 17-keypoint format.

## Pipeline

```text
Video -> Frames -> YOLO-Pose -> 17 x (x,y,confidence) -> SAM ground truth
                                      ^
                                      |
CSI JSON -> clean/group -> 3x53 -> bursts -> timestamp synchronization
                                      |
                                      v
                    CNN+FC / CNN+GRU / Transformer
                                      |
                                      v
                          Predicted SAM -> 17 joints
```

## Important changes for this actual dataset

- 1 Tx x 3 Rx x 64 raw complex CSI.
- Remove indices 27..37, giving 53 usable CSI positions per Rx.
- Saved complete snapshot rate is around 48 Hz, not assumed 100 Hz.
- CSI is bursty; typical burst contains about 5 complete snapshots.
- Video is around 30 FPS.
- Synchronization uses `metadata.video.start_unix` and CSI `host_ts`.
- CPU YOLO annotation is optimized: by default only frames nearest to CSI burst targets are inferred, reducing the number of pose inferences substantially.
- `000_empty_room` is skipped by YOLO-Pose and may be used only for optional CSI calibration.

## Default teacher

`yolo26n-pose.pt`, CPU, COCO-17. If `models/yolo26n-pose.pt` exists it is used; otherwise Ultralytics receives the model name and can download the pretrained weight.

## Default model

`MODEL_NAME = "cnn_gru_sam"`

Other options: `cnn_fc_sam`, `transformer_sam`.

## Loss

`L = L_coordinate + 0.2 * L_relative_SAM`, confidence-weighted by YOLO-Pose keypoint confidence.

## Run

See `commands.txt`.

## Changes, results, and sitting failure analysis

See [docs/CHANGES_AND_RESULTS.md](docs/CHANGES_AND_RESULTS.md) for:

- what was added (empty class, live inference, P1/P2 experiment stack);
- measured LOSO results (especially subject 001);
- why sitting predictions are still wrong.

## Cross-day CSI variation (day1 / day2 / day3)

See [docs/CROSS_DAY_CSI_VARIATION.md](docs/CROSS_DAY_CSI_VARIATION.md) for:

- what **corr** means between CSI fingerprints across days;
- measured sitting / standing / empty day-to-day differences;
- why live inference fails under that shift;
- how to train and evaluate to overcome it.

## Project status (completed + currently running)

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for:

- everything completed (archive, 3-day layout, sync, docs);
- current train/val/test and empty-room split;
- what is running now and what to run next.

## Results: day3_id007 evaluation + empty-room protocol

See [docs/RESULTS_DAY3_ID007.md](docs/RESULTS_DAY3_ID007.md) for:

- pose / classification numbers for experiment `day3_id007`;
- detailed empty-room failure analysis (day1 OK, day2/day3 fail);
- suggested empty-room evaluation checklist before live inference.

## Results: day4_id005_emptyfix (current best)

See [docs/RESULTS_DAY4_ID005_EMPTYFIX.md](docs/RESULTS_DAY4_ID005_EMPTYFIX.md) for:

- pose / classification on test person 005 (days 1–4);
- empty detection fixed to 100% across day1–day4;
- comparison vs `day3_id007` and live inference commands.

## Final output (model + config + how to run)

See [docs/FINAL_OUTPUT.md](docs/FINAL_OUTPUT.md) for:

- saved `.pt` path under `models/`;
- full configuration of the delivered model;
- metrics summary and live/eval commands.

