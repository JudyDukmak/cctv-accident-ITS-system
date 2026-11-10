| std_seed0 | fold 0 | configs/videomae_v1.json (sha256 894d3436) | scheme B, standard aug | best epoch 0 | val T@1 0.4408 | within 2 s 0.488 | early stop after epoch 3 | diagnostics: results/diagnostics/std_seed0_fold0 |
| B_v2_seed0 | fold 0 only (no folds 1, 2) | configs/videomae_v2_B.json (sha256 5ee05e2f) | scheme B, standard aug, validation every 620 updates | best check 1 (epoch 1.00) | val T@1 0.4400 | within 2 s 0.487 | early stop after check 7 | diagnostics: results/diagnostics/B_v2_seed0_fold0 |
| C_v2_seed0 | folds 0, 1, 2 | folds 0, 2: configs/videomae_v2_C.json (sha256 1edaaedf); fold 1: videomae_v2_C_rtx4050.json (sha256 fe4f55dc), documented deviation | scheme C, standard aug, validation every 620 updates | best checks 2 / 2 / 6 (epoch 0.67 / 0.65 / 1.36) | val T@1 0.5828 / 0.6151 / 0.8518; pooled out-of-fold 0.6773 (2211 clips) | within 2 s 0.658 / 0.673 / 0.902; pooled 0.739 | early stop after checks 8 / 8 / 12 | results/classifier_summary.md; results/diagnostics/C_v2_seed0_3folds; results/diagnostics/C_v2_seed0_fold0, _fold1, _fold2 |
| C_cos3_seed0 | planned: folds 0, 1, 2 | folds 0, 2 (RTX 5060): configs/videomae_v2_C_cos3.json (sha256 56998505); fold 1 (RTX 4050): configs/videomae_v2_C_cos3_rtx4050.json (sha256 3fc8bd38) | scheme C, standard aug, cosine over 3 epochs, warmup 0.5 epoch, validation every 620 updates | - | - | - | - | not trained yet |
| C_cctv_seed0 | folds 0, 1, 2 | folds 0, 2 (RTX 5060): configs/videomae_v2_C_cctv.json (sha256 c890b480); fold 1 (RTX 4050): configs/videomae_v2_C_cctv_rtx4050.json (sha256 31e3e00c), documented deviation as C_v2 | scheme C, cctv_v2 aug, night protection dark_luma 50, validation every 620 updates (Ablation 1) | best checks 2 / 3 / 6 (epoch 0.67 / 0.87 / 1.36) | val T@1 0.6032 / 0.5986 / 0.8564; pooled out-of-fold 0.6802 (2211 clips; C_v2 0.6773) | within 2 s 0.686 / 0.662 / 0.910; pooled 0.747 | early stop after checks 8 / 9 / 12 | DECISION (pre-registered rule, higher pooled T@1): C_cctv; +0.0029, scenario bootstrap [-0.0147, +0.0167]: no measurable difference on synthetic validation. results/classifier_summary.md; results/diagnostics/C_cctv_vs_C_v2; _fold0, _fold1, _fold2; preparation: runs/aug_preview/C_v2_cctv, _cctv_v2, _cctv_v2_night50 |

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

C_cctv_seed0 fold 0, first launch 2026-11-10 06:46 (git 665e26d, dirty): crashed on the first training batch with
UnboundLocalError (row_weather) in the per-weather counting of protected windows: the weather map was built only for
the old "cctv" night guard, not for cctv_v2. No optimizer update, no checkpoint, no validation, no result. Fixed in
train_classifier.py (the weather map is built whenever night_guard is set); augmentation and training unchanged.

C_cctv_seed0 partial (folds 0, 2), first analysis 2026-11-10, NO decision (fold 1 not analysed; the ablation is
decided on pooled out-of-fold T@1 over all three folds, Section 6 of the training document). Folds 0 and 2 trained
on the RTX 5060 from git a66ce3d (dirty: only results/diagnostics/std_seed0_fold0/report.txt and untracked
preview / luma / smoke-test files; train_classifier.py and its imports equal a66ce3d and were not modified during
the runs). Integrity vs C_v2_seed0 on the same folds: stored configs differ only in augmentation and night_guard;
label, split and cache hashes equal; the cctv_v2 profile in config.json equals CCTV_V2. Same best checks as C_v2
(2 and 6). Paired on the same 1456 clips (folds 0+2): T@1 0.7225 vs 0.7096 (+0.0130; clip bootstrap 95%
[+0.0017, +0.0243], optimistic because clips of a scenario are not independent; scenario-cluster bootstrap
[-0.0059, +0.0272]); within 2 s 0.791 vs 0.773. Protected training windows: fold 0 94.1% of night windows loaded,
fold 2 100% (its training folds contain no Town03_head-on night clip), 0% of non-night. Details:
results/diagnostics/C_cctv_vs_C_v2_folds02/report.txt (compare_folds.py --paired).

C_cctv_seed0, all 3 folds, Ablation 1 decision (2026-11-10). Fold 1: RTX 4050, Python 3.14.0, git a66ce3d (dirty;
the dirty file list is not recorded on that machine), micro_batch 4 x grad_accum 4 as C_v2 fold 1; stored config
differs from C_v2 fold 1 only in augmentation and night_guard; labels and split hashes equal; cache c76c46e6 = the
documented RTX 4050 manifest (results/diagnostics/C_v2_seed0_3folds/fold1_cache_manifest_rtx4050.json). Fold 1 was
resumed twice from step_latest.pt at update 4617 (after check 6): augmentation draws after a resume come from new
worker generators, and its protected-window counts restart at the resume. Best check reproduces exactly (0.5986);
the last check (9) rescored on the RTX 5060 gives 0.4995 vs 0.4985 logged (+0.0010, cross-GPU bf16, as C_v2 fold 1).
Pooled out-of-fold T@0.5 / 1 / 2 / within 2 s: C_v2 0.5728 / 0.6773 / 0.7477 / 0.7390; C_cctv 0.5627 / 0.6802 /
0.7519 / 0.7472. Rule: final configuration = higher pooled out-of-fold T@1 -> C_cctv (0.6802 > 0.6773); difference
+0.0029, clip bootstrap [-0.0063, +0.0124] (optimistic), scenario-cluster bootstrap (14 scenarios) [-0.0147,
+0.0167]: no measurable difference on synthetic validation. Next: seeds 1 and 2 of C_cctv (C_cctv_seed1,
C_cctv_seed2). Details: results/diagnostics/C_cctv_vs_C_v2/ (report.txt, decision.txt).

## Post-processing (frozen on C_cctv_seed0 pooled out-of-fold)

Frozen 2026-11-10 on synthetic validation only (no real data, no training, CPU): CHOICE = smoothing width 3 (centred
moving average, zero-padded edges as in training) + parabolic peak interpolation. Parabola: with smoothed scores
s-1, s0, s+1 around the peak, delta = 0.5 (s-1 - s+1) / (s-1 - 2 s0 + s+1), clipped to [-0.5, 0.5] windows, 0 at clip
edges or if the denominator >= 0; t_hat = t0 + delta * 0.5 s. Confidence unchanged (peak minus highest smoothed score
outside +-2 windows).
Rule (stated before the results): primary = pooled out-of-fold T@1 of C_cctv_seed0 over 2211 clips; the best of the
6 variants (width 1, 3, 5 x interpolation none, parabola) replaces the default (3, none) only if it beats it by >= 0.002;
T@0.5 reported separately; C_v2_seed0 reference only.
Numbers (C_cctv_seed0, best checks): default (3, none) T@0.5 / 1 / 2 = 0.5627 / 0.6802 / 0.7519 (reproduces the saved
predictions in 2211/2211 clips); (3, parabola) 0.5768 / 0.6854 / 0.7534, gain +0.0052 T@1 (+0.0141 T@0.5); per fold
T@1 0.6085 / 0.6053 / 0.8598 (default 0.6032 / 0.5986 / 0.8564); within 2 s 0.7458 (default 0.7472); median |error|
0.378 s (0.438); Spearman rho(confidence, |error|) -0.375 (-0.377). Other variants: (1, none) 0.6251, (1, parabola)
0.6473, (5, none) 0.6493, (5, parabola) 0.6586. Reference C_v2_seed0: (3, none) 0.6773 -> (3, parabola) 0.6815.
Applied unchanged to seeds 1 and 2 and to the real test: aggregate_runs.py ... --smooth 3 --interp parabola (the
default call keeps the saved predictions). Checkpoint selection during training is unchanged (best check by val T@1
with the default rule); post-processing is applied to the best-check window scores afterwards. The choice was made on
the same pooled out-of-fold predictions it is reported on, so +0.0052 is optimistic. Details:
results/diagnostics/postproc_C_cctv_seed0/ (report.txt, postproc_variants.csv).

Flip test-time augmentation (2026-11-10, synthetic validation only, RTX 5060): ADOPTED. TTA score per window = mean of
the sigmoid scores of the original window and of the window with all 16 frames horizontally flipped (same
normalisation), then smoothing 3 + parabola as above. Rule (stated before the results): adopt only if pooled
out-of-fold T@1 over 2211 clips improves by >= 0.002 over no TTA, both arms scored with best.pt on the same GPU;
T@0.5 / T@2 reported, not deciding. Reproduction of the original-frame scores vs the saved best-check predictions:
folds 0 and 2 exact (t_hat 770/770 and 686/686, T@1 equal); fold 1 (trained on the RTX 4050) t_hat equal in
749/755 clips, T@1 0.5974 vs 0.5986 (-0.0012, cross-GPU bf16). Pooled no TTA -> flip TTA: T@0.5 0.5767 -> 0.5826,
T@1 0.6850 -> 0.6922 (+0.0072 >= 0.002), T@2 0.7532 -> 0.7612, within 2 s 0.7454 -> 0.7544, median |error| 0.379 ->
0.378 s, Spearman rho(confidence, |error|) -0.375 -> -0.381. Per fold T@1: 0.6085 -> 0.6241, 0.6043 -> 0.5941
(fold 1 worse), 0.8598 -> 0.8767. T@1 by weather: clear +0.012, sunset +0.006, wet +0.004, rain +0.005, night +0.008;
by type: head-on +0.013, rear-end +0.008, sideswipe +0.012, single -0.031 (n 66), t-bone -0.002. Chosen on the same
pooled predictions it is reported on, so the gain is optimistic. Cost (classifier only, bf16, val batch 16, incl.
host-to-device copy; data loading overlapped by 4 workers): one pass 595.9 ms per clip (18.6 ms per window, mean 32
windows per clip), two passes 1191.5 ms per clip (37.3 ms per window), i.e. 2x. Applied unchanged to seeds 1 and 2
and to the real test: analyse_predictions.py --score-best --hflip --out-dir DIR, then aggregate_runs.py ...
--scores-dir DIR --tta flip --smooth 3 --interp parabola. Details: results/diagnostics/postproc_C_cctv_seed0/tta/
(tta_report.txt, tta_table.csv, tta_by_group.csv, fold_<k>/ scores and scoring_report.txt).
