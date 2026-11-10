| std_seed0 | fold 0 | configs/videomae_v1.json (sha256 894d3436) | scheme B, standard aug | best epoch 0 | val T@1 0.4408 | within 2 s 0.488 | early stop after epoch 3 | diagnostics: results/diagnostics/std_seed0_fold0 |
| B_v2_seed0 | fold 0 only (no folds 1, 2) | configs/videomae_v2_B.json (sha256 5ee05e2f) | scheme B, standard aug, validation every 620 updates | best check 1 (epoch 1.00) | val T@1 0.4400 | within 2 s 0.487 | early stop after check 7 | diagnostics: results/diagnostics/B_v2_seed0_fold0 |
| C_v2_seed0 | folds 0, 1, 2 | folds 0, 2: configs/videomae_v2_C.json (sha256 1edaaedf); fold 1: videomae_v2_C_rtx4050.json (sha256 fe4f55dc), documented deviation | scheme C, standard aug, validation every 620 updates | best checks 2 / 2 / 6 (epoch 0.67 / 0.65 / 1.36) | val T@1 0.5828 / 0.6151 / 0.8518; pooled out-of-fold 0.6773 (2211 clips) | within 2 s 0.658 / 0.673 / 0.902; pooled 0.739 | early stop after checks 8 / 8 / 12 | results/classifier_summary.md; results/diagnostics/C_v2_seed0_3folds; results/diagnostics/C_v2_seed0_fold0, _fold1, _fold2 |
| C_cos3_seed0 | planned: folds 0, 1, 2 | folds 0, 2 (RTX 5060): configs/videomae_v2_C_cos3.json (sha256 56998505); fold 1 (RTX 4050): configs/videomae_v2_C_cos3_rtx4050.json (sha256 3fc8bd38) | scheme C, standard aug, cosine over 3 epochs, warmup 0.5 epoch, validation every 620 updates | - | - | - | - | not trained yet |
| C_cctv_seed0 | planned: folds 0, 1, 2 | folds 0, 2 (RTX 5060): configs/videomae_v2_C_cctv.json (sha256 c890b480); fold 1 (RTX 4050): configs/videomae_v2_C_cctv_rtx4050.json (sha256 31e3e00c) | scheme C, cctv_v2 aug, night protection dark_luma 50, validation every 620 updates (Ablation 1) | - | - | - | - | not trained yet; preparation: runs/aug_preview/C_v2_cctv, _cctv_v2, _cctv_v2_night50 |

Config sha256 = canonical config hash as recorded in runs/<run>/fold_<k>/config.json (first 8 hex digits).

C_v2_seed0, fold 1 (documented deviation, accepted): trained on an RTX 4050 (Python 3.14.0) with micro_batch 4 x
grad_accum 4 instead of 8 x 2 (effective batch 16 in every fold; no other config key differs). Its frame cache was
rebuilt on that machine: manifest results/diagnostics/C_v2_seed0_3folds/fold1_cache_manifest_rtx4050.json (sha256
c76c46e6, = recorded) equals ours in every field except created_at, skipped_this_run, written_this_run; 5 of 5
sampled .npz files byte-identical. The fold-1 check-8 checkpoint gives val T@1 0.5501 on the RTX 4050 and 0.5498
on the RTX 5060. Aggregate with: aggregate_runs.py C_v2_seed0 --allow-config-diff micro_batch,grad_accum
--run-cache-manifest C_v2_seed0:1=results\diagnostics\C_v2_seed0_3folds\fold1_cache_manifest_rtx4050.json

C_cos3_seed0 (planned): videomae_v2_C.json with only max_epochs 10 -> 3 and warmup_fraction 0.05 -> 0.16664; the
RTX 4050 config additionally micro_batch 4, grad_accum 4. warmup_fraction 0.16664 means 0.5 epoch for these folds:
warmup 1398 / 1428 / 1595 of 8391 / 8568 / 9570 updates (folds 0 / 1 / 2), the same warmup as C_v2. It is not a
general rule: recompute it if max_epochs, the batch or the split change.

C_cctv_seed0 (planned, Ablation 1: CCTV-style augmentation on top of C_v2): videomae_v2_C.json with only
augmentation "standard" -> "cctv_v2" (since 2026-11-10; first planned as "cctv", see "Profile v2" below) and the
new key night_guard {"dark_luma": 17.9}; the RTX 4050 config additionally
micro_batch 4, grad_accum 4. Same folds, seed, schedule (updates/epoch, warmup, total, checks = C_v2: 2797/1398/27970/46,
2856/1428/28560/47, 3190/1595/31900/52 for folds 0/1/2) and loss weights. Preparation logs, unit test, luma tables,
previews and loader benchmark: runs/aug_preview/C_v2_cctv.

Ablation 1 augmentation order, as implemented (decision B, 2026-11-09, before any cctv result was seen): frame-rate
drop -> horizontal flip -> blur -> low resolution -> colour shift -> JPEG. The Section 6 table of
Training_Configuration_Classifier.docx lists the flip last; the code order is kept because flipping the scene before
simulating the camera matches the physical pipeline (scene -> capture -> colour -> compression). The random draws
are identical in both orders; only pixel rounding differs (mean 0.13 grey levels in differing windows, max 39, mainly
from JPEG). Details: runs/aug_preview/C_v2_cctv/ORDER_NOTE.txt.

Night guard (rule chosen by the author: never darken windows that are already dark): if a window's mean luma
(0.299 R + 0.587 G + 0.114 B, before the colour shift) is below dark_luma, the colour shift draws brightness from
[1.0, 1.3] instead of [0.5, 1.3] (the same uniform draw mapped to the narrower range; draw sequence unchanged).
dark_luma 17.9 = unaugmented night median of 300 synthetic split_dev-train clips (60 per weather, seed 0;
night_luma_stats.py); PROVISIONAL until confirmed: changing it changes the config hash. Absent key = previous
behaviour; hashes of all existing configs verified unchanged.

Profile v2 ("cctv_v2", decided 2026-11-10, before any cctv run; no cctv result exists, so no result was seen). The
"cctv" profile above stays in the code unchanged but is not used for C_cctv_seed0. Reasons:
- label-preserving principle: an augmented window whose crash is not visible to a person turns the label into
  noise; the visual check of the "cctv" previews showed many such windows (stacked degradations, strong downscale,
  night windows);
- 50% of windows got 3 or more of 5 degradations (Binomial(5, 0.5); measured 0.498 of 2000 fold-0 windows);
- the downscale basis was wrong: every clip, synthetic or real, is resized to 224 x 224 (full frame), so a 314p real
  clip keeps full 224 detail; 30-70% of 224 is harsher than any real resolution.
cctv_v2: each of frame-rate drop, blur, low resolution, colour shift, JPEG with p 0.3, at most 2 per window (random
seeded choice of 2 if more are drawn); hflip unchanged (p 0.5, after the frame-rate drop); fps {4,5,6}, blur sigma
[0.3, 1.0], downscale [0.5, 0.9] of 224, brightness [0.7, 1.3], contrast [0.8, 1.2], saturation [0.7, 1.3], hue
+-0.02, grayscale p 0.1 in the colour branch, JPEG quality [25, 50]. Dark windows (unaugmented mean luma < dark_luma
17.9): at most 1 degradation, brightness [1.0, 1.3], contrast [1.0, 1.2]. One draw per window for all 16 frames;
validation never augmented. The v2 ranges are fixed in train_classifier.py (CCTV_V2, also written to each run's
config.json) because the config differs from videomae_v2_C.json only in augmentation and night_guard; cctv_aug is
not used by cctv_v2. Measured: 0 / 1 / 2 degradations in 0.176 / 0.402 / 0.422 of windows; night windows below the
unaugmented night p1: 23 (= unaugmented; "cctv" 137). The two C_cctv configs were replaced (no run used them; old
canonical hashes f2655d2a / aafcf87d). Details: runs/aug_preview/C_v2_cctv/PROFILE_V2_NOTE.txt; logs, previews,
tables and benchmark: runs/aug_preview/C_v2_cctv_v2.

Night protection at dark_luma 50 (2026-11-10, before any cctv run; no cctv result exists). Reason: the visual check
showed that night windows just above dark_luma 17.9 (luma 18.4 and 23.4, with downscale 0.5 + brightness 0.7) are
hard to read. dark_luma = 50 (chosen by the student) protects dark and medium-lit night windows; well-lit night
windows (luma >= 50) keep the normal cctv_v2 rule; night augmentation is reduced, not removed. Protected windows
(unaugmented luma < 50): at most 1 degradation, brightness [1.0, 1.3], contrast [1.0, 1.2], downscale [0.7, 0.9],
blur sigma [0.3, 0.7], JPEG quality [35, 50]; fps drop, saturation, hue and grayscale unchanged; the same single draw
is mapped to the narrower range (no extra draws). Non-protected windows are pixel-identical to cctv_v2 at 17.9. In
the 300-clip sample no non-night window is below 50 (lowest 89.6); 93.4% of night windows are protected, and all
night windows with luma >= 50 are from Town03_head-on. Night windows with 0 / 1 / 2 degradations: 0.161 / 0.809 /
0.030 (17.9: 0.161 / 0.595 / 0.245); day unchanged. Configs: only night_guard.dark_luma 17.9 -> 50 (canonical
922ea6fa -> c890b480, 201b2da4 -> 31e3e00c). Details: runs/aug_preview/C_v2_cctv_v2/PROFILE_V2_NOTE.txt; logs,
tables, previews and benchmark: runs/aug_preview/C_v2_cctv_v2_night50.
