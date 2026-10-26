# Preprocessing record: synthetic training data

This record covers the four synthetic-data preprocessing steps, in pipeline order. All numbers come from the summary files listed in each section; copies of those summaries are in `logs/preprocessing/`. No real-data file is used by any step described here.

Every hash below was recomputed on 2026-10-26 while writing this record and matches the value stored in `split_manifest.json` or `cache/frames224_v1/cache_manifest.json`.

Decisions are labelled either **source: <document/section>** or **design choice**.

---

## 1. Synthetic split (`make_synthetic_split.py`)

**Purpose.** The script assigns each of the 2,211 synthetic clips to one of three validation folds. Clips are grouped by scenario (map + accident type), so the same scripted crash never appears in both training and validation. The split is frozen by a SHA-256 hash so that it cannot change silently.

**Commands**
```
py make_synthetic_split.py
py make_synthetic_split.py --verify        # prints OK when the file matches split_sha256
```

**Inputs and outputs**
- Input: `ACCIDENT/dataset/metadata-synthetic.csv`. Only this file is read; no video is opened.
- Output: `synthetic_split.csv`, with columns `rgb_path, scenario_id, val_fold, split_dev, map, type, weather, camera_position, split_version`.
- Output: `split_manifest.json`.
- Output: `synthetic_check/split_summary.txt`.
- The previous version (v1) is archived unchanged in `archive/split_v1_handmade/`.

**Key numbers** (from `synthetic_check/split_summary.txt` and `split_manifest.json`)

The split version is `v2_rebalanced`, with 2,211 clips and 14 scenarios.

| val_fold | scenarios | val clips | train clips | val share | val types |
|---|---|---|---|---|---|
| 0 | Town04_rear-end, Town04_sideswipe, Town06_head-on, Town10HD_single, Town10HD_t-bone | 770 | 1441 | 34.8% | rear-end 245, head-on 202, sideswipe 139, t-bone 131, single 53 |
| 1 | Town03_t-bone, Town05_head-on, Town05_rear-end, Town05_sideswipe, Town07_single | 755 | 1456 | 34.1% | rear-end 277, t-bone 184, head-on 174, sideswipe 107, single 13 |
| 2 | Town03_head-on, Town06_sideswipe, Town07_rear-end, Town07_t-bone | 686 | 1525 | 31.0% | rear-end 272, head-on 212, sideswipe 159, t-bone 43, single 0 |

All 19 EXPECTED lines in the summary are OK.

**Hash**
- `split_sha256` = `965e1104208e3e7e53e64ebcac8f8d467c6647797c706560fb71fe503ad8fb19`
- `frozen: true`

**Decisions**

**Scenario-level grouping** (`scenario_id = map + "_" + type`).
- Source: `check_synthetic_runs.py` (outputs `synthetic_check/run_check.csv` and `synthetic_check/sim_tick_summary.csv`) and `inspect_synthetic_events.py`.
- The trailing number in a clip name is the camera position, not an event id.
- Clips of one scenario are separate re-runs of the same scripted crash at the same location: the median distance between crash points is 4.5 m across weather and camera.
- So a per-clip split would leak crash sites between training and validation.

**Same-map scenarios are separate crash sites, 76–758 m apart.**
- Source: `make_scenario_table.py` (outputs `synthetic_check/scenario_table.csv`, `scenario_table.md`).

**Fold assignment v2 (cost 13.06 → 7.40).**
- Source: Week 3 evaluation, Section 5.4. The search is a random search with seed 0 and 200,000 draws, and the best scenario-level result is kept. It is implemented in `rebalance_folds.py`, with output in `synthetic_check/rebalance_report.txt`.
- Cost of v1 (hand-made): 13.06.
- Cost of v2: 7.395, made of size 1.02 + timing 1.375 + type penalty 5.0. The type penalty of 5.0 is the minimum possible, because only two scenarios are "single".
- The global median used by the cost is the median of the 14 scenario medians, 6.675 s.
- Effect: the spread of the label-1 share across validation folds falls from 34.5 to 5.3 percentage points (`synthetic_check/fold_balance_check.txt`).

**Map-level split rejected.**
- Source: `split_manifest.json` `grouping_rule`, which was checked by scoring all 540 map-level splits.
- The best map-level split costs 12.71, with a timing term of 6.75 and one fold whose median accident time is 11.5 s. The scenario-level split costs 7.395.

**Every fold contains head-on, rear-end, sideswipe and t-bone.**
- Design choice, enforced as a hard check.
- Fold 2 contains no "single" clips, because only two single scenarios exist.

**`split_dev` = fold 0 as the default quick split.** Design choice.

**Freeze guard.** Design choice. A run that would produce a different file stops unless `--unfreeze` is given.

---

## 2. Sliding windows (`make_windows.py`)

**Purpose.** The script cuts every synthetic clip into fixed-length, overlapping windows of 16 frames sampled at 8 fps, which is the input format of the video model. Each window gets a label (scheme A) relative to the accident time, plus a phase (pre, gap, onset or post). The script refuses to run if the split file does not match its frozen hash.

**Command**
```
py make_windows.py --overwrite
```
It was rerun after the v2 split. `--overwrite` replaced the v1 windows, which remain archived in `archive/split_v1_handmade/`.

**Inputs and outputs**
- Input: `synthetic_split.csv`. Its SHA-256 is checked against `split_sha256` before any window is built.
- Input: `ACCIDENT/dataset/metadata-synthetic.csv`.
- Output: `synthetic_windows.csv`, with columns `rgb_path, scenario_id, val_fold, split_dev, window_idx, t_start, t_end, frame_first, frame_last, t_star, label, phase, split_version`. Rows are sorted by `(rgb_path, window_idx)`.
- Output: `windows_config.json`. Its content was unchanged by the rerun.
- Output: `synthetic_check/windows_summary.txt`.
- Output: `windows_sha256`, added to `split_manifest.json`.

**Key numbers** (from `synthetic_check/windows_summary.txt`)
- **Frame rate:** all 2,211 clips are 20.0 fps (`no_frames / duration`).
- **Frame/time consistency:** 0 clips have |accident_frame / fps − accident_time| > 0.026 s.
- **Total windows:** 70,730.
- **Labels (scheme A):** −1 (gap) 4,392 (6.2%); 0 (pre) 22,073 (31.2%); 1 (onset + post) 44,265 (62.6%).
- **Phases:** pre 22,073; gap 4,392; onset 8,113; post 36,152.
- **Per validation fold** (label −1 / 0 / 1):
  - fold 0: 1,531 / 7,709 / 16,748
  - fold 1: 1,507 / 7,664 / 15,870
  - fold 2: 1,354 / 6,700 / 11,647
- **Windows per clip:** min 8, median 32, max 61.
- **Accident time:** min 1.05 s, median 6.90 s, max 20.75 s.
- **Clips with no label-0 window:** 62 (the crash is too early). Clips with no label-1 window: 0.
- **EXPECTED checks:** all 12 are OK (`Town05_sideswipe_rain_44`: 36 windows; `Town05_sideswipe_clear_00`: 38 windows).

**Hashes**
- `windows_sha256` = `36866a3009097928654957f48bb88bd0a3565a8853ef8439da4fc81a67d1abad` (in `split_manifest.json`)
- `windows_config.json` SHA-256 = `f0f45d3bb7a7820efea3dda97bfc0c132c6d1acc54f69488edbdccec5a8222e7`. This hash is recomputed for this record; it is not stored in any manifest.

**Decisions**

**16 frames per window.** Source: VideoMAE-base configuration (`MCG-NJU/videomae-base-finetuned-kinetics`, `num_frames: 16`).

**8 fps sampling.** Design choice. It uses the same value as `MODEL_FPS = 8` in `audit_metadata.py`.

**Stride 0.5 s.** Design choice.

**Window span.** (16 − 1) / 8 = 1.875 s. Window j has `t_start = 0.5·j`, sample times `t_i = t_start + i/8`, and `t_end = t_start + 1.875`.

**Window centre** = `t_start + 0.9375` (half the span). It is stored as `t_centre` in step 3.

**Frame index** = `floor(t · fps_video + 0.5)`, with `fps_video = no_frames / duration`.
- Source for the fps: the ACCIDENT baselines convert frames and seconds with `no_frames / duration` (`optical_flow.py:170`, `bbox_dynamics.py:330`).
- The round-half-up rule (rather than Python's banker's rounding) is a design choice. The product is rounded to 9 decimals to remove float noise.

**A window is kept while its last frame index ≤ no_frames − 1.** Design choice.

**Label rule A, with t* = accident_time and margin m = 1.0 s:**
- `t_end < t* − m` gives 0 (pre)
- `t* − m ≤ t_end < t*` gives −1 (gap, ignored)
- `t_end ≥ t*` and `t_start < t*` gives 1 (onset)
- `t_start ≥ t*` gives 1 (post)

This is a design choice. The 1.0 s margin is the same as `M` in `alarm_metrics.py` and the "t* − 1 s" boundary in the evaluation, Section 6.3. Window times are rounded to 6 decimals, so that values like 9.55 − 1.0 compare exactly.

---

## 3. Window labels A / B / C (`make_window_labels.py`)

**Purpose.** The script adds two alternative labelling schemes to the windows of step 2 without changing them. Scheme B marks only the windows that contain the impact as positive, and scheme C gives soft weights that fall off with distance from the impact. Scheme A is copied unchanged for comparison.

**Commands**
```
py make_window_labels.py --overwrite
py make_window_labels.py --record-hash     # stores window_labels_sha256 in split_manifest.json
```
The manifest's `window_labels_sha256` equals the current SHA-256 of `synthetic_window_labels.csv`.

**Inputs and outputs**
- Input: `synthetic_windows.csv`. Its SHA-256 is checked against `windows_sha256` first.
- Inputs: `split_manifest.json`, `windows_config.json`. The half span (0.9375 s) and the margin (1.0 s) are checked against the config.
- Output: `synthetic_window_labels.csv`, with columns `rgb_path, scenario_id, val_fold, window_idx, t_start, t_end, t_centre, t_star, label_A, label_B, weight_C_text, weight_C_code, split_version`. Rows are in the same order as the windows.
- Output: `synthetic_check/window_labels_summary.txt`.

**Key numbers** (from `synthetic_check/window_labels_summary.txt`)
- **Rows:** 70,730.
- **label_B:** −1: 40,313; 0: 22,073; 1: 8,344.
- **label_B per validation fold** (−1 / 0 / 1):
  - fold 0: 15,388 / 7,709 / 2,891
  - fold 1: 14,527 / 7,664 / 2,850
  - fold 2: 10,398 / 6,700 / 2,603
- **label_B = 0 matches label_A = 0** with 0 mismatches.
- **label_B = 1 windows:** 8,344. These are the 8,113 onset windows plus the 231 windows with `t_start = t*`, and they are the same rows.
- **Positives per clip under B:** min 1, median 4, max 4. No clip has zero positives.
- **weight_C_text:**
  - exactly 1.0: 8,344
  - exactly 0.0: 22,182. This is the 22,073 pre windows plus 109 far-aftermath windows, where the Gaussian underflows to 0 at d ≥ 19.35 s.
  - strictly between 0 and 1: 40,204
- **weight_C_code:** exactly 1.0: 8,344; exactly 0.0: 109; strictly between: 62,277.
- **The two weights differ on 22,073 windows**, all of which have `t_end < t* − 1`, i.e. exactly the pre windows.
- **All EXPECTED checks for `Town05_sideswipe_rain_44` are OK.** For example, window 16 has `t_centre` 8.9375. `weight_C_text` is 0, 0.4020 and 0.9406 for windows 13, 14 and 15. `weight_C_code` is 0.0632 and 0.0037 for windows 13 and 12.

**Hash**
- `window_labels_sha256` = `4487770a0d15531a2d8f37357428daa70513f4704e53b96af5fca9c570340e11` (in `split_manifest.json`)

**Decisions**

**`t_centre = t_start + 0.9375`.** Design choice; see step 2.

**`label_A`** is copied unchanged from step 2.

**Label rule B.** Source: evaluation, Section 6.3.
- `t_end < t* − 1.0` gives 0.
- `t_start ≤ t* ≤ t_end` gives 1.
- Everything else (margin or aftermath) gives −1, which is ignored in the loss.

**`weight_C_text`.** Source: evaluation, Section 6.3 text and code.
- 0.0 if `t_end < t* − 1.0`.
- 1.0 if the window contains t*.
- Otherwise `exp(−d² / (2 · 0.5²))`, with `d = min(|t* − t_start|, |t* − t_end|)`.
- The 0 for pre windows follows the text: "In all schemes, windows ending before t* − 1 s are normal."

**`weight_C_code`.** Source: evaluation, Section 6.3 code, taken literally.
- 1.0 if the window contains t*, otherwise the same Gaussian, with no rule for pre windows.
- A window ending exactly at t* − 1 s therefore gets exp(−2) = 0.135.

**The evaluation document is inconsistent between its text and its code.** The choice between the two versions is pending confirmation. The difference is small:
- Only 246 pre windows have `weight_C_code` > 0.1, and the largest value is 0.1223.
- Summed over all pre windows, the code version adds about 125.5 positive weight, against 8,344 from the windows that contain t*. (Measured during development; not stored in a summary file.)

**σ = 0.5 s for the Gaussian.** Source: evaluation, Section 6.3 code.

**Boundary `t_end < round(t* − 1.0, 6)`.** Design choice. It uses the same rounding as step 2, so that label_B = 0 exactly where label_A = 0.

---

## 4. Frame cache (`build_frame_cache.py`)

**Purpose.** The script decodes every synthetic video once and stores the frames that windows actually use, resized to the model's 224 × 224 input, as JPEGs. Training then reads small per-clip files instead of decoding 1920 × 1080 video for every window. Every window is a contiguous 16-frame slice of a clip's cached frame grid.

**Commands** (training environment)
```
.venv-ml\Scripts\python.exe build_frame_cache.py --limit 5     # smoke test
.venv-ml\Scripts\python.exe build_frame_cache.py               # full run (2 workers, the default)
```

**Inputs and outputs**
- Input: `synthetic_windows.csv`. Its SHA-256 is checked against `windows_sha256`.
- Inputs: `split_manifest.json`, `windows_config.json`, `ACCIDENT/dataset/metadata-synthetic.csv`.
- Input: the synthetic videos in `ACCIDENT/dataset/synthetic_videos/videos/`.
- Output: `cache/frames224_v1/<clip_stem>.npz`, one per clip, with arrays `k` (int32), `frame_idx` (int32), `offsets` (int64, length K+1) and `data` (uint8 JPEG bytes). `cache/` is git-ignored.
- Output: `cache/frames224_v1/cache_manifest.json` and `cache/frames224_v1/cache_summary.txt`.
- Reader: `read_cached_frame(npz, i)` returns the RGB frame at grid position i.

**Key numbers** (from `cache_summary.txt` and `cache_manifest.json`)
- **Clips:** 2,211 of 2,211 present; `complete: true`; 0 errors. 2,206 clips were written by the full run; the 5 clips from the smoke test were skipped because they already existed with the right grid.
- **Frames:** 309,452.
- **Size:** 6,713,850,124 bytes (6.253 GiB).
- **Build time:** 17.30 minutes.
- **Decoder:** PyAV 19.0.2, sequential, no seeking. OpenCV 5.0.0 was used for resizing, encoding and verification.
- **Verification** (first 5 clips; grid positions 0, K/2 and K−1):
  - The cached JPEG was compared with the same frame decoded again by OpenCV (sequentially) and resized the same way.
  - The mean absolute pixel difference is 1.405–2.031 on a 0–255 scale.
  - In all 15 cases the cached frame matches its own frame index better than the neighbouring indices ±1.

**Hashes**
- `cache_manifest.json` records `windows_sha256` = `36866a3009097928654957f48bb88bd0a3565a8853ef8439da4fc81a67d1abad`, the windows file the cache was built from.
- The individual `.npz` files are not hashed.

**Decisions**

**Frame grid** `t_k = k / 8`, `k = 0..K−1`, frame index `floor(t_k · no_frames/duration + 0.5)`, `K = 8 · max(t_start) + 16`.
- Design choice. It makes every window equal to `grid[8·t_start : 8·t_start + 16]`.
- This is checked for every window against `make_windows.window_frame_indices` before decoding, and the largest grid index is checked to be ≤ no_frames − 1.

**224 × 224 full-frame resize without crop (squash).**
- Source for 224: VideoMAE-base configuration (`image_size: 224`).
- Source for "no crop": `synthetic_check/crash_size_summary.txt`. A short-side resize followed by a central square crop would put the crash box centre outside the crop in 75 of 2,211 clips (3.4%) and cut the box partly in 188 clips (8.5%), with up to 19.0% of clips losing the box centre in Town03_t-bone.
- Trade-off, from the same file: squashing shrinks the crash box. At 224 its median long side is 30.5 px (40.7 px with a short-side resize), and 6.9% of boxes fall below 16 px (2.6% with a short-side resize).

**`cv2.INTER_AREA` for downscaling; frames stored as RGB.** Design choice.

**JPEG quality 95.** Design choice. The verification difference (MAD ≈ 1.4–2.0 / 255) includes the JPEG loss.

**Sequential decoding with PyAV, no seeking.** Design choice. Frame indices then count decoded frames from frame 0, which is the same convention as the windows.

**Temporary file plus rename; restart skips finished clips.** Design choice. It protects against partial files after an interrupted run.

---

## Data integrity checks

Every check below passed.

| Step | Check | Result | Output saved in |
|---|---|---|---|
| 1 | `rgb_path` unique in the synthetic metadata | passed (hard check) | `synthetic_check/split_summary.txt` (the run completes only if it passes) |
| 1 | Exactly the 14 scenario ids, each in exactly one fold; every row assigned | passed (hard check) | `synthetic_check/split_summary.txt` |
| 1 | Every fold contains head-on, rear-end, sideswipe, t-bone | passed (hard check) | `synthetic_check/split_summary.txt` |
| 1 | EXPECTED fold sizes and type counts | 19 / 19 OK | `synthetic_check/split_summary.txt` |
| 1 | Freeze guard: a different split is refused without `--unfreeze` | passed (tested on scratch copies) | terminal only |
| 1 | `synthetic_split.csv` matches `split_sha256` (`--verify`) | OK; recomputed again for this record | `split_manifest.json` |
| 1 | v1 and v2 label-1 shares reproduce known values (75.5/68.3/41.0 and 64.4/63.4/59.1 %) | OK | `synthetic_check/fold_balance_check.txt` |
| 2 | Split SHA-256 checked before building windows | matches | `synthetic_check/windows_summary.txt` |
| 2 | Split ↔ metadata merge is 1:1 on 2,211 rows | passed (hard check) | `synthetic_check/windows_summary.txt` |
| 2 | fps_video = 20.0 for all clips; accident frame/time consistent (0 clips > 0.026 s) | passed | `synthetic_check/windows_summary.txt` |
| 2 | Windows consecutive per clip; labels only in the order 0 → −1 → 1; no label-1 window with t_end < t*; no label-0 window with t_end ≥ t* − 1; ≥ 1 window per clip | passed (hard checks) | `synthetic_check/windows_summary.txt` |
| 2 | EXPECTED windows/labels for two reference clips | 12 / 12 OK | `synthetic_check/windows_summary.txt` |
| 2 | `windows_config.json` unchanged by the v2 rerun | identical SHA-256 | terminal only |
| 2 | v2 windows equal v1 windows except fold columns (70,730 rows; 0 differences in times, frames, labels, phases) | passed | terminal only |
| 2 | `val_fold` in windows equals `synthetic_split.csv` | 0 differing clips | `synthetic_check/fold_balance_check.txt` |
| 3 | Windows SHA-256 checked before labelling; config half span and margin match | passed | `synthetic_check/window_labels_summary.txt` |
| 3 | label_B = 0 exactly where label_A = 0 | 0 mismatches | `synthetic_check/window_labels_summary.txt` |
| 3 | label_B = 1 = onset windows + windows with t_start = t* | 8,113 + 231 = 8,344, same rows | `synthetic_check/window_labels_summary.txt` |
| 3 | No clip without a positive under B | 0 clips | `synthetic_check/window_labels_summary.txt` |
| 3 | weight_C_text ≠ weight_C_code only on pre windows | 22,073 windows, all pre | `synthetic_check/window_labels_summary.txt` |
| 3 | EXPECTED label_B / t_centre / weight_C values for `Town05_sideswipe_rain_44` | all OK | `synthetic_check/window_labels_summary.txt` |
| 3 | Manifest update changes no other key | passed (re-read check) | `split_manifest.json` |
| 4 | Windows SHA-256 checked before caching | matches | `cache/frames224_v1/cache_summary.txt` |
| 4 | Every window equals its grid slice; max grid index ≤ no_frames − 1 | passed for all 2,211 clips | `cache/frames224_v1/cache_summary.txt` |
| 4 | All clips decoded without error; cache complete | 2,211 / 2,211, 0 errors, complete true | `cache/frames224_v1/cache_manifest.json` |
| 4 | Cached frames match an independent re-decode at the correct index | MAD 1.405–2.031; correct index best in 15 / 15 | `cache/frames224_v1/cache_summary.txt` |

Copies of the four step summaries and of `crash_size_summary.txt` are kept unchanged in `logs/preprocessing/`.

---

## Limitations

- **Map sharing across folds.** The folds are grouped by scenario, not by map. Town04, Town05, Town07 and Town10HD have scenarios in different folds. Validation therefore measures generalisation to new crash sites, which can include a town seen during training, not to new towns.
- **Only 14 independent scenarios.** Results will vary strongly from fold to fold. Fold 2 has no "single" clips, and Town07_single has only 13 clips.
- **The scheme C weighting is not final.** The evaluation's Section 6.3 text and code disagree on windows ending before t* − 1 s. Both versions are stored until this is confirmed.
- **The squash resize shrinks small crashes.** At 224 × 224, 6.9% of crash boxes have a long side below 16 px.
- **The crash-site distances are estimates.** The 76–758 m figure comes from projecting collision boxes onto the ground plane for 104 sampled clips, and most of them used a default ground height of 0.
- **The cache verification is a spot check.** It covers 5 clips at 3 positions each. In the middle of a clip, the advantage over the neighbouring frames is small, because consecutive frames of static scenes are similar.
- **The cache files are not individually hashed.** Only the windows file they were built from is recorded.
- **Some checks were shown only in the terminal**: the freeze-guard test, the v1/v2 window comparison and the `windows_config.json` hash comparison.
- **62 clips have no label-0 window**, because their crash happens within the first ~2 s.
- **Real-data timing is not covered here.** The `accident_frame`/`accident_time` disagreement found in 321 real clips (64 in the frozen test set) concerns real-data evaluation, not this synthetic pipeline.
