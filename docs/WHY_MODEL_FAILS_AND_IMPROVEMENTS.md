# Why the Model Fails — and What to Improve

This note explains the failure modes seen on this CSI pose project (especially **live inference** and **new empty captures**), and a practical improvement plan.

Related results:

- Day4 success (in-domain): `docs/RESULTS_DAY4_ID005_EMPTYFIX.md`, `docs/FINAL_OUTPUT.md`
- Cross-day CSI analysis: `docs/CROSS_DAY_CSI_VARIATION.md`
- Review notes: `suggestions.md`

---

## 1. What works vs what fails

| Situation | Typical result | Meaning |
|-----------|----------------|---------|
| Replay `data/raw/csi/day5/...` with day5-trained model | Often correct (empty / sit / stand) | Model + preprocessing OK **in training domain** |
| Offline test person 005 (same days as train) | ~87% PCK@0.2 | Pose generalizes somewhat to held-out **people** |
| Held-out empties from day1–5 **folder** sessions | Often ~100% empty | Empty works **inside those days’ RF** |
| Live BPI stream (same hardware, “today”) | Often **always standing** | Live CSI ≠ training CSI distribution |
| Root / OOD empties (`011_empty_room.json`, `012_…`, loose `000_…`) | Sitting/standing, `P(empty)≈0` | Unseen RF empty fingerprint |
| Day4 model on day5 JSON | Wrong poses/classes | Cross-day domain shift |

**Honest status**

- Research / offline in-domain pipeline: **strong**
- Reliable live demo under drifting WiFi conditions: **not solved by architecture alone**

---

## 2. Root cause (not a drawing bug)

### 2.1 RF domain mismatch

CSI is a fingerprint of the **radio environment**, not only “person pose”:

- room multipath, interference, channel, traffic pattern  
- tiny setup drift even when antennas “look” unchanged  
- day-to-day change (day4 model fails on day5)

The classifier learns:

> “CSI that looks like empties / sits / stands **I saw in training**”

not a perfect physics rule for “nobody is here.”

When live CSI is out of distribution, softmax often collapses to **standing** (strongest / most stable class).

### 2.2 Empty is especially brittle

Empty has **no joints** — only a class label.  
If empty training covers only certain days’ `000_empty_room*` sessions, a new empty capture looks like a person.

Raising empty loss weight (×20) fixed **in-domain** empty recall; it cannot invent coverage for RF never seen in train.

### 2.3 Sitting ≈ standing in RF

Within one day, sitting and standing CSI can be highly correlated.  
Under domain shift, sit→stand errors increase. Sitting PCK (~79%) stays harder than standing (~96%).

### 2.4 GUI / checkpoint mismatch (operational failure)

The live GUI loads **only**:

```text
pose_detection_live/best_model.pt
```

It does **not** auto-load `outputs/checkpoints/..._best.pt`.

Failure mode we hit:

1. Retrain day5 / L1L2 successfully  
2. Forget to copy into `pose_detection_live/best_model.pt`  
3. GUI still runs **day4** weights → “I added today’s data but live is still wrong”

**After every train:**

```bash
cp outputs/checkpoints/<MODEL>_<TAG>_best.pt \
   pose_detection_live/best_model.pt
# then restart the GUI
```

Also ensure CSI norm matches the tag:

```text
outputs/csi_normalization_stats_<EXPERIMENT_TAG>.npz
```

(`pose_engine` / live pipeline read this via `src.config`.)

### 2.5 What L1/L2 does and does not do

AdamW L2 + L1 (`WEIGHT_DECAY`, `L1_WEIGHT` in `src/config.py`) reduce **weight overfitting**.

They do **not** fix live RF domain shift. New live empty/sit/stand data still required.

### 2.6 SAM / jointsMatrix

We use a SAM `[2,17,17]` similar to Wi-Pose `jointsMatrix` (absolute diagonal + relative off-diagonal).  
That helps pose structure; it is not a live-domain fix.

---

## 3. How to diagnose quickly

```text
1) Replay day5 empty / sit / stand JSON with current GUI checkpoint
      OK  → model+GUI OK for training domain
      BAD → wrong .pt or wrong CSI norm (check epoch in GUI log)

2) Live, room empty
      P(empty) high → domain OK enough to demo
      P(empty)≈0, standing → live RF OOD (expected failure)

3) Save live empty JSON from BPI, replay that file
      Still wrong → must train on that capture
      Replay OK but stream wrong → investigate live buffering (rarer)
```

---

## 4. What you need to improve (priority)

### Must (highest impact)

1. **Always deploy the checkpoint you just trained** into `pose_detection_live/best_model.pt` and restart the GUI.  
2. **Collect paired data on the exact live demo setup**: empty + sitting + standing, same day, same traffic/client.  
3. Put most new empties in `EMPTY_TRAIN_SESSIONS`; hold 1–2 out for test (do not train on all and call them “unseen”).  
4. Rebuild split/norm if needed, retrain with a **new** `EXPERIMENT_TAG`, evaluate, then copy `.pt` to live.  
5. Treat live as a **domain test**, not as proof that offline PCK is enough.

### Should

- Session-level metrics: false-skeleton time on empty, worst-session PCK — not only window-level PCK@0.2.  
- Live gating: consecutive votes / hysteresis; suppress skeleton when uncertain.  
- Wire YOLO annotate to use `config.py` thresholds for reproducible labels.  
- Audit hard sitting YOLO labels (knees/ankles).  
- Keep multi-day empty in train forever; never go back to single-day empty only.

### Nice later (after live domain improves)

- Binary presence head (empty vs occupied), then sit/stand.  
- TCN / deeper GRU / Transformer ablations **one at a time**.  
- Direct 34-coordinate head with SAM as auxiliary.  
- Runtime empty calibration (20–30 s confirmed empty at demo start).

### Do not waste time on (alone)

- Bigger model before new live-domain data  
- Empty class weight ≫ 20  
- Tuning only on flattering day-folder empties  
- Expecting L1/L2 or TCN to fix “always standing” live  

---

## 5. Recommended workflow after each collection day

```bash
# 1) Put paired CSI/video under data/raw/csi/dayN and videos/dayN
# 2) Update EMPTY_*_SESSIONS + EXPERIMENT_TAG in src/config.py
# 3) Frames + YOLO (new day only) → rebuild dataset if needed
.venv/bin/python -m src.extract_frames --day dayN
.venv/bin/python -m src.annotate_pose --day dayN
.venv/bin/python -m src.build_dataset

# 4) Split + train + eval
.venv/bin/python -c "from src.dataset import create_split_indices; create_split_indices(force=True)"
.venv/bin/python -m src.train
.venv/bin/python -m src.evaluate

# 5) Deploy to GUI (mandatory)
cp outputs/checkpoints/cnn_gru_multitask_sam_${TAG}_best.pt \
   pose_detection_live/best_model.pt
# restart live_gui.py
```

---

## 6. One-sentence summary

**The model is not “broken at drawing skeletons”; it memorizes training RF domains, so live and unseen empties fail until those conditions are in training — and the GUI only improves after you copy the new checkpoint into `pose_detection_live/best_model.pt`.**
