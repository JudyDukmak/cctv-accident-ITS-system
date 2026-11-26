| std_seed0 | fold 0 | configs/videomae_v1.json (sha256 894d3436) | scheme B, standard aug | best epoch 0 | val T@1 0.4408 | within 2 s 0.488 | early stop after epoch 3 | diagnostics: results/diagnostics/std_seed0_fold0 |
| B_v2_seed0 | fold 0 only (no folds 1, 2) | configs/videomae_v2_B.json (sha256 5ee05e2f) | scheme B, standard aug, validation every 620 updates | best check 1 (epoch 1.00) | val T@1 0.4400 | within 2 s 0.487 | early stop after check 7 | diagnostics: results/diagnostics/B_v2_seed0_fold0 |
| C_v2_seed0 | folds 0, 1, 2 | folds 0, 2: configs/videomae_v2_C.json (sha256 1edaaedf); fold 1: videomae_v2_C_rtx4050.json (sha256 fe4f55dc), documented deviation | scheme C, standard aug, validation every 620 updates | best checks 2 / 2 / 6 (epoch 0.67 / 0.65 / 1.36) | val T@1 0.5828 / 0.6151 / 0.8518; pooled out-of-fold 0.6773 (2211 clips) | within 2 s 0.658 / 0.673 / 0.902; pooled 0.739 | early stop after checks 8 / 8 / 12 | results/classifier_summary.md; results/diagnostics/C_v2_seed0_3folds; results/diagnostics/C_v2_seed0_fold0, _fold1, _fold2 |
| C_cos3_seed0 | folds 0, 1, 2 | configs/videomae_v2_C_cos3.json (sha256 56998505) in all three folds, all on the RTX 5060 (micro_batch 8 x 2; fold 1 used videomae_v2_C_cos3.json, not the rtx4050 config); git cd9aba7 (dirty) | scheme C, standard aug, cosine over 3 epochs, warmup 0.5 epoch, validation every 620 updates | best checks 2 / 2 / 6 (epoch 0.66 / 0.65 / 1.36) | val T@1 0.5825 / 0.6170 / 0.8539; pooled out-of-fold 0.6785 (2211 clips; C_v2 0.6773) | within 2 s 0.658 / 0.675 / 0.902; pooled 0.740 | early stop after checks 8 / 8 / 12; fold 1 resumed 3 times at update 4340, best check before | NOT ADOPTED by the pre-registered 0.02 rule (+0.0012 pooled T@1 over C_v2). results/diagnostics/C_cos3_seed0_3folds. An earlier RTX 4050 fold-1 run (videomae_v2_C_cos3_rtx4050.json, 3fc8bd38, best val T@1 0.6164) lies outside runs\ (Senior 2\C_cos3_seed0\fold_1) and is not used |
| C_cctv_seed0 | folds 0, 1, 2 | folds 0, 2 (RTX 5060): configs/videomae_v2_C_cctv.json (sha256 c890b480); fold 1 (RTX 4050): configs/videomae_v2_C_cctv_rtx4050.json (sha256 31e3e00c), documented deviation as C_v2 | scheme C, cctv_v2 aug, night protection dark_luma 50, validation every 620 updates (Ablation 1) | best checks 2 / 3 / 6 (epoch 0.67 / 0.87 / 1.36) | val T@1 0.6032 / 0.5986 / 0.8564; pooled out-of-fold 0.6802 (2211 clips; C_v2 0.6773) | within 2 s 0.686 / 0.662 / 0.910; pooled 0.747 | early stop after checks 8 / 9 / 12 | DECISION (pre-registered rule, higher pooled T@1): C_cctv; +0.0029, scenario bootstrap [-0.0147, +0.0167]: no measurable difference on synthetic validation. results/classifier_summary.md; results/diagnostics/C_cctv_vs_C_v2; _fold0, _fold1, _fold2; preparation: runs/aug_preview/C_v2_cctv, _cctv_v2, _cctv_v2_night50 |
| C_cctv_seed1 | folds 0, 1, 2 | as C_cctv_seed0 (c890b480 / 31e3e00c / c890b480), seed 1; folds 0, 2 git 80bac63 (dirty), fold 1 a66ce3d (dirty) | scheme C, cctv_v2 aug, night protection dark_luma 50, validation every 620 updates (final configuration, seed 1) | best checks 2 / 2 / 12 (epoch 0.67 / 0.65 / 2.53) | val T@1 0.5840 / 0.6099 / 0.8528; pooled out-of-fold 0.6762 (2211 clips); frozen post-processing 0.6854 | within 2 s 0.662 / 0.685 / 0.911; pooled 0.747 | early stop after checks 8 / 8 / 18; fold 2 resumed once at update 5500, best check after the resume | results/classifier_summary.md; results/diagnostics/C_cctv_seeds012; C_cctv_seed1_fold0, _fold1, _fold2 |
| C_cctv_seed2 | folds 0, 1, 2 | as C_cctv_seed0 (c890b480 / 31e3e00c / c890b480), seed 2; folds 0, 2 git 80bac63 (dirty), fold 1 a66ce3d (dirty) | scheme C, cctv_v2 aug, night protection dark_luma 50, validation every 620 updates (final configuration, seed 2) | best checks 2 / 2 / 12 (epoch 0.67 / 0.65 / 2.53) | val T@1 0.5869 / 0.6121 / 0.8699; pooled out-of-fold 0.6833 (2211 clips); frozen post-processing 0.6923 | within 2 s 0.668 / 0.672 / 0.926; pooled 0.749 | early stop after checks 8 / 8 / 18; fold 1 resumed once at update 5500, best check before the resume | results/classifier_summary.md; results/diagnostics/C_cctv_seeds012; C_cctv_seed2_fold0, _fold1, _fold2 |

Config sha256 = canonical config hash as recorded in runs/<run>/fold_<k>/config.json (first 8 hex digits).

C_v2_seed0, fold 1 (documented deviation, accepted): trained on an RTX 4050 (Python 3.14.0) with micro_batch 4 x
grad_accum 4 instead of 8 x 2 (effective batch 16 in every fold; no other config key differs). Its frame cache was
rebuilt on that machine: manifest results/diagnostics/C_v2_seed0_3folds/fold1_cache_manifest_rtx4050.json (sha256
c76c46e6, = recorded) equals ours in every field except created_at, skipped_this_run, written_this_run; 5 of 5
sampled .npz files byte-identical. The fold-1 check-8 checkpoint gives val T@1 0.5501 on the RTX 4050 and 0.5498
on the RTX 5060. Aggregate with: aggregate_runs.py C_v2_seed0 --allow-config-diff micro_batch,grad_accum
--run-cache-manifest C_v2_seed0:1=results\diagnostics\C_v2_seed0_3folds\fold1_cache_manifest_rtx4050.json

C_cos3_seed0 (planned; trained since, see its table row): videomae_v2_C.json with only max_epochs 10 -> 3 and warmup_fraction 0.05 -> 0.16664; the
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

Ablation 1 under the frozen post-processing (2026-11-12; REPORTING ONLY, the decision above is unchanged; synthetic
validation, no real data, no training). C_v2_seed0 best.pt scored on the RTX 5060 with analyse_predictions.py
--score-best --hflip (fold 1 with the RTX 4050 cache manifest): folds 0 and 2 reproduce the saved predictions
exactly; fold 1 t_hat equal in 746/755 clips, T@1 0.6168 vs 0.6151 (+0.0017, cross-GPU bf16). Aggregated with
aggregate_runs.py --tta-study --smooth 3 --interp parabola (flip TTA arm; the same recomputed_preds(..., tta="flip")
path as --tta flip, without overwriting results/classifier_summary). Cells: T@0.5 / T@1 / T@2 / within 2 s; pooled =
out-of-fold over 2211 clips; C_cctv mean +- sample std over seeds 0-2 (ddof 1, n 3); C_v2 has one seed.

| post-processing | run | fold 0 | fold 1 | fold 2 | pooled |
|---|---|---|---|---|---|
| (a) default | C_v2 seed 0 | 0.4631 / 0.5828 / 0.6700 / 0.6584 | 0.5227 / 0.6151 / 0.6912 / 0.6728 | 0.7512 / 0.8518 / 0.8972 / 0.9023 | 0.5728 / 0.6773 / 0.7477 / 0.7390 |
| (a) default | C_cctv seed 0 | 0.4734 / 0.6032 / 0.6869 / 0.6857 | 0.4878 / 0.5986 / 0.6807 / 0.6623 | 0.7453 / 0.8564 / 0.9032 / 0.9096 | 0.5627 / 0.6802 / 0.7519 / 0.7472 |
| (a) default | C_cctv seeds 0-2 | 0.4637 +- 0.0086 / 0.5914 +- 0.0104 / 0.6748 +- 0.0105 / 0.6719 +- 0.0123 | 0.4932 +- 0.0122 / 0.6069 +- 0.0073 / 0.6893 +- 0.0086 / 0.6728 +- 0.0113 | 0.7483 +- 0.0107 / 0.8597 +- 0.0090 / 0.9078 +- 0.0084 / 0.9155 +- 0.0089 | 0.5620 +- 0.0077 / 0.6799 +- 0.0036 / 0.7521 +- 0.0012 / 0.7478 +- 0.0010 |
| (b) frozen | C_v2 seed 0 | 0.4868 / 0.5992 / 0.6773 / 0.6662 | 0.5347 / 0.6164 / 0.6914 / 0.6649 | 0.7758 / 0.8763 / 0.9170 / 0.9242 | 0.5928 / 0.6910 / 0.7565 / 0.7458 |
| (b) frozen | C_cctv seed 0 | 0.5021 / 0.6241 / 0.7044 / 0.6987 | 0.4937 / 0.5941 / 0.6758 / 0.6543 | 0.7710 / 0.8767 / 0.9191 / 0.9271 | 0.5826 / 0.6922 / 0.7612 / 0.7544 |
| (b) frozen | C_cctv seeds 0-2 | 0.4863 +- 0.0140 / 0.6043 +- 0.0171 / 0.6833 +- 0.0183 / 0.6766 +- 0.0192 | 0.5085 +- 0.0173 / 0.6128 +- 0.0163 / 0.6915 +- 0.0138 / 0.6733 +- 0.0171 | 0.7667 +- 0.0173 / 0.8710 +- 0.0110 / 0.9145 +- 0.0092 / 0.9208 +- 0.0085 | 0.5809 +- 0.0103 / 0.6900 +- 0.0040 / 0.7578 +- 0.0034 / 0.7512 +- 0.0028 |

Pooled T@1, C_cctv - C_v2: (a) seed 0 +0.0029, seed mean +0.0026; (b) seed 0 +0.0012, seed mean -0.0011. Both
within one C_cctv seed std (0.0036 / 0.0040) and C_v2 has one seed: no measurable difference under either
post-processing, as decided. The frozen post-processing raises C_v2 by +0.0137 pooled T@1 (0.6773 -> 0.6910), C_cctv
seed 0 by +0.0120. T@0.5 is lower for C_cctv under both (pooled, seed mean - C_v2: -0.0108 (a), -0.0120 (b)), mostly
from fold 1. Details: results/diagnostics/C_v2_seed0_frozen/ (ablation_table.txt/.csv, frozen/ and smooth3_none/ tta
reports, scores/fold_<k>/scoring_report.txt, commands.txt). Figure: results/figures/F3_ablation_bars.png.

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

## Final configuration: mean +- std over seeds

C_cctv, seeds 0, 1, 2 (2026-11-12; synthetic 3-fold validation only, no real data; scoring on the RTX 5060, no
training; nothing tuned). Integrity of seeds 1 and 2: config hash c890b480 (folds 0, 2; RTX 5060; git 80bac63, dirty;
before this update the only tracked file differing from 80bac63 was results/diagnostics/std_seed0_fold0/report.txt)
and 31e3e00c (fold 1; RTX 4050; git a66ce3d,
dirty; cache c76c46e6 = the documented RTX 4050 manifest); train_classifier.py and its imports are identical in a66ce3d
and 80bac63; seed recorded; label and split hashes and the stored config equal seed 0's per fold; every fold ends
with "training finished" (early stopping). Resumes: seed 1 fold 2 once at update 5500 (8 checks done, last at 4960);
its best check 12 (update 8060) comes AFTER the resume, so the selected model saw augmentation draws from new worker
generators. Seed 2 fold 1 once at update 5500; its best check 2 (update 1860) comes before the resume. Reproduction:
best and last checks reproduce metrics.csv exactly on folds 0 and 2; fold 1 best checks exact, last checks +0.0012 /
+0.0028 (seed 1 / 2, scored on the RTX 5060, cross-GPU bf16). best.pt rescored on the RTX 5060: folds 0, 2 exact;
fold 1 t_hat equal in 752/755 (T@1 equal) and 750/755 (T@1 -0.0013) clips.
Pooled out-of-fold over 2211 clips, T@0.5 / T@1 / T@2 / within 2 s, mean +- sample std (ddof 1, n 3):
(a) default (smoothing 3, no interpolation, no TTA; saved predictions): seeds 0 / 1 / 2 T@1 0.6802 / 0.6762 /
0.6833; 0.5620 +- 0.0077 / 0.6799 +- 0.0036 / 0.7521 +- 0.0012 / 0.7478 +- 0.0010.
(b) frozen (smoothing 3 + parabola + flip TTA): seeds 0 / 1 / 2 T@1 0.6922 / 0.6854 / 0.6923; 0.5809 +- 0.0103 /
0.6900 +- 0.0040 / 0.7578 +- 0.0034 / 0.7512 +- 0.0028.
Per fold T@1 (a) 0.5914 +- 0.0104 / 0.6069 +- 0.0073 / 0.8597 +- 0.0090; (b) 0.6043 +- 0.0171 / 0.6128 +- 0.0163 /
0.8710 +- 0.0110 (folds 0 / 1 / 2).
Independent check of the frozen post-processing on seeds 1 and 2 (not used for the choice): (b) - (a) pooled T@1
+0.0091 / +0.0090, T@0.5 +0.0157 / +0.0208, T@2 +0.0036 / +0.0044, within 2 s +0.0027 / +0.0005: positive on both
seeds for every metric; per fold T@1 positive in 6 of 6 (seed 0's fold-1 drop is not repeated); within 2 s per fold
negative in 2 of 6.
Seed-to-seed range of best-check T@1 (a): fold 0 0.0193, fold 1 0.0135, fold 2 0.0171, next to the seed-0 C_cctv -
C_v2 differences +0.0204 / -0.0165 / +0.0046: slightly above the C_cctv seed range on folds 0 and 1, inside it on
fold 2, with opposite signs; the per-fold Ablation 1 differences are of the size of seed variation (C_v2 has one
seed, so its own seed variation is not included), consistent with "no measurable difference".
Spearman rho(confidence, |error|) with (b): pooled -0.381 / -0.460 / -0.419 (seeds 0 / 1 / 2), negative in every fold
and seed (weakest fold 2 of seed 0, -0.119); no confidence threshold chosen.
Reported synthetic result of the final configuration: (b) pooled T@1 0.6900 +- 0.0040 (n 3 seeds).
results/classifier_summary.{csv,md} holds (a) (the default aggregate); (b) is in results/diagnostics/C_cctv_seeds012/
(seed_tables.txt, per_seed_fold.csv, mean_std_over_seeds.csv, independent_postproc_check.csv, seed_range_T1.csv,
confidence_rho_frozen.csv, integrity.txt, seed{s}_frozen/ and seed{s}_smooth3_none/ tta reports, commands.txt).

## Routing threshold tau: rule (frozen), value 0.01 (computed 2026-11-16, see "Zero-shot VLM full scan" below)

Evidence (synthetic validation only, CPU, no real data; tau NOT computed): confidence_vs_error.py --tau-evidence
on results/diagnostics/C_cctv_seeds012/seed{0,1,2}_frozen/tta_clip_predictions_flip.csv (2211 clips per seed, 6633
pooled). p_narrow(c) = P(|t_hat - t*| <= 2 s | confidence c): increasing isotonic regression (pool adjacent
violators, ties in confidence pooled) on the 6633 pooled predictions, evaluated on c = 0.00 ... 0.96 by linear
interpolation between the fitted values (results/diagnostics/tau_evidence/isotonic_curve.csv). p_narrow at c = 0.1 /
0.2 / 0.3 / 0.4 / 0.5: 0.6023 / 0.7311 / 0.8532 / 0.8768 / 0.9357. Cumulative, pooled: confidence >= 0.5 covers
0.1842 of the predictions with 0.9673 within 2 s; >= 0.3 covers 0.4156 with 0.9267; >= 0.1 covers 0.7259 with
0.8440; all 0.7512. Caveats: the three seeds give three predictions per clip, so the pooled predictions are not
independent (the Wilson intervals are too narrow); above c = 0.8 there are only 69 predictions; the curve is
fitted on the same pooled out-of-fold predictions it describes. Details: results/diagnostics/tau_evidence/
(tau_evidence.txt, tau_cumulative.csv, tau_bins.csv, isotonic_curve.csv, isotonic_fit_points.csv); figure
results/figures/F9_confidence_evidence.png.

Rule (frozen 2026-11-14, before p_full is known): tau = the smallest confidence c with p_narrow(c) >= p_full, where
p_narrow is the isotonic curve above (C_cctv seeds 0-2, pooled out-of-fold, frozen post-processing) and p_full is
the share of synthetic validation clips whose full-scan VLM time is within 2 s, measured in the zero-shot VLM stage
on synthetic clips only. Above tau the narrow path is expected to be at least as accurate as the full scan and is
cheaper. The value is computed once, when p_full is known.

Expected routing on real clips (2026-11-17; cost estimate only, tau stays 0.01; no real label read: only rgb_path,
confidence and fallback of results/real_test/predictions/C_cctv_seed{0,1,2}/clip_predictions.csv and the path column
of real_test_set.csv). Clips with confidence < 0.01 go to the full scan; the 3 no-window clips have confidence 0 and
count as below.

| Clips | Seed 0 | Seed 1 | Seed 2 | Mean over seeds |
|---|---|---|---|---|
| All 2027 | 203 (0.1001) | 265 (0.1307) | 227 (0.1120) | 231.67 (0.1143) |
| Frozen 400 | 40 (0.1000) | 58 (0.1450) | 58 (0.1450) | 52.00 (0.1300) |

Details: results/real_test/routing_count/ (routing_count.py, command.txt, report.txt).

### Routing rule replaced: tau on T@1 (2026-11-21; change made on synthetic data only, before the real cascade run)

On the supervisor's request (feedback of 2026-11-18), the within-2-s rule above (tau = 0.01) is replaced before any
real cascade run. Synthetic data only; no model run; no real clip or label used to decide anything. Command:
`.venv-ml\Scripts\python.exe confidence_vs_error.py --routing-t1 --out-dir results\routing_T1`. Inputs: C_cctv seeds
0-2 out-of-fold predictions (frozen post-processing), the frozen full-scan v3 results (results/vlm_full_scan) and the
clip midpoint (duration / 2), all 2211 clips; per-clip T@1 = exp(-(t - t*)^2 / 2).

Fallback comparison, pooled (6633 predictions), mean T@1 on the same clips:

| Confidence | n | Classifier | Full scan | Midpoint |
|---|---|---|---|---|
| [0, 0.01) | 310 | 0.3047 | 0.1269 | 0.1654 |
| [0.01, 0.02) | 243 | 0.3712 | 0.1850 | 0.1798 |
| [0.02, 0.05) | 549 | 0.4142 | 0.1228 | 0.1957 |
| [0.05, 0.1) | 716 | 0.5206 | 0.1459 | 0.1909 |
| [0.1, 0.2) | 1147 | 0.6066 | 0.1671 | 0.2079 |
| [0.2, 1] | 3668 | 0.8440 | 0.1945 | 0.2230 |

Fallback (fixed rule: confidence < 0.05, pooled, higher mean T@1): full scan 0.1377, midpoint 0.1837 (n 1102) ->
**fallback = clip midpoint**. Rule: tau = the smallest c (grid 0.00 ... 0.96) where the classifier's isotonic T@1
(increasing, pooled 6633) >= the midpoint's isotonic T@1 against the same confidence. Result: **tau = 0.00**
(classifier 0.1774 vs midpoint 0.0031 at c = 0.00; the classifier's curve stays above at every grid point), so all
synthetic predictions (6633/6633; 2211/2211 per seed) keep the classifier t_hat. Real clips, from the confidence
column only: 0 of the frozen 400 (and 0 of 2027) below tau for each seed; the 3 no-window clips already get
t_hat = duration / 2 by the frozen no-window fallback. The full scan is not used for timing in the cascade.
Details: results/routing_T1/ (report.txt, fallback_by_confidence.csv with per-seed tables, isotonic_curves_T1.csv,
isotonic_fit_points_T1.csv, isotonic_curves_T1.png, real_frozen400_below_tau.txt, command.txt).

## Real test (run once)

Window classifier alone on the real clips (no VLM, no tau), 2026-11-14/15, RTX 5060 Laptop GPU, no training. Nothing
was tuned or chosen on these results. Lock results/real_test/RUN_ONCE.txt, created 2026-11-14T22:09:43Z: git HEAD
a93bcf1; aggregate_runs.py, analyse_predictions.py and build_frame_cache.py were modified after a93bcf1 (decision-2
truncation, run-once lock, real evaluation) and are uncommitted, so their sha256 values in the lock are the record of
the code that ran (equal again after stage C2). The lock holds the sha256 of 7 code files, real_windows.csv, the cache
manifest, metadata-real.csv and real_test_set.csv (checked before every scoring run and evaluation), and the config
hash, best check and best.pt sha256 of the 12 fold models. Stage C1 (C_v2_seed0, C_cctv_seed0) was evaluated at
2026-11-15T03:18:14Z, then stage C2 (C_cctv_seed1, C_cctv_seed2) at 08:24:12Z, each once and with nothing changed
between them. The prediction sha256 values are in the lock's EVENT lines.

Real frame cache (Part A, CPU) used the synthetic rules: 8 fps grid, 224 x 224 full-frame resize, no crop, JPEG 95,
PyAV sequential decoding. It is in cache/frames224_real_v1/ (real_windows.csv sha256 fa1d2c07...). Of 2027 clips,
2024 have at least one window (399 of the 400); there are 82132 windows (16375 in the 400) and 0 errors. Windows are
16 frames with stride 0.5 s, t_centre = t_start + 0.9375 s and fps = no_frames / duration; the window function
reproduces synthetic_windows.csv exactly. The grid formula is make_windows.frame_index
(floor(round(t * fps, 9) + 0.5)); the unrounded synthetic-cache formula would differ by one frame in 14 real clips
(94 grid positions), and we kept the rounded one (decision 2026-11-15).

Decisions of 2026-11-15, made before any real label was read:
(1) The 3 clips with no window (3gY2nZW6ZCM_00, which is in the 400; MsQns-Oy-hk_02; jYai-crKZx8_00) get
t_hat = duration / 2 and confidence 0, and count in every metric. Results are also reported on the clips with at
least one window.
(2) Three files have fewer decodable frames than metadata no_frames (none is in the 400). Each keeps
fps = no_frames / duration and only the windows whose frames exist:

| Clip | no_frames vs decodable | Windows | Dropped window_idx |
|---|---|---|---|
| LZY4HepPGu0_00 | 274 vs 240 | 57 -> 49 | 49-56 |
| YQxqeP2Q_-M_00 | 352 vs 348 | 57 -> 56 | 56 |
| lgU3T6kySLw_00 | 181 vs 180 | 57 -> 56 | 56 |

(3) Per run, the predictor averages the flip-TTA window scores of the 3 fold best.pt models, then applies
smoothing 3 + parabola (Week 3 evaluation 6.5). This ensemble has no synthetic validation. Each fold model alone is
reported as a secondary result, and nothing is chosen from it.

Code check (Part B, synthetic) ran one fold model through the real-test path on its own validation clips
(C_cctv_seed0, folds 0 and 2). It reproduced the saved frozen predictions: 0 window scores differ, t_hat is equal in
770/770 and 686/686 clips, and T@1 is equal.

Results are T@0.5 / T@1 / T@2 / within 2 s, with C_cctv as mean +- sample std over seeds 0-2 (ddof 1):

| Scope | C_cctv seeds 0-2 | C_v2 seed 0 | Naive midpoint |
|---|---|---|---|
| All 2027 (headline) | 0.2351 +- 0.0073 / 0.3934 +- 0.0077 / 0.5679 +- 0.0072 / 0.5562 +- 0.0080 | 0.2655 / 0.4245 / 0.5908 / 0.5836 | 0.1068 / 0.1896 / 0.2952 / 0.2797 |
| Frozen 400 | 0.2422 +- 0.0094 / 0.4019 +- 0.0040 / 0.5724 +- 0.0032 / 0.5533 +- 0.0076 | 0.2588 / 0.4210 / 0.5919 / 0.5800 | 0.1078 / 0.1932 / 0.2956 / 0.2850 |
| At least 1 window (2024) | 0.2343 +- 0.0073 / 0.3926 +- 0.0077 / 0.5673 +- 0.0072 / 0.5555 +- 0.0080 | 0.2647 / 0.4238 / 0.5902 / 0.5830 | n/a |
| Early impact, t* < 2.875 s (460) | 0.2982 +- 0.0061 / 0.4731 +- 0.0072 / 0.6334 +- 0.0026 / 0.6297 +- 0.0033 | 0.3379 / 0.5124 / 0.6634 / 0.6565 | 0.1688 / 0.2764 / 0.3875 / 0.3870 |

The naive midpoint reproduces the paper's 0.1068 / 0.1896 / 0.2952.

Ablation 1 on real: C_v2_seed0 has a higher T@1 than each C_cctv seed, both on all 2027 clips (0.4245 vs
0.3976 / 0.3981 / 0.3846) and on the frozen 400 (0.4210 vs 0.3980 / 0.4060 / 0.4017). C_v2 has only one seed, and on
synthetic the two variants did not differ measurably. The C_cctv selection is not changed on test results.

Transfer gap (real 2027 minus synthetic pooled out-of-fold):

| Metric | C_cctv seeds 0-2 | C_v2_seed0 |
|---|---|---|
| T@0.5 | -0.3457 +- 0.0176 | n/a |
| T@1 | -0.2966 +- 0.0104 | -0.2665 |
| T@2 | -0.1899 +- 0.0058 | n/a |
| Within 2 s | -0.1951 +- 0.0054 | n/a |

Confidence on real (C_cctv seeds 0-2 pooled, 6081 predictions) is much lower than on synthetic:

| | Real | Synthetic |
|---|---|---|
| Median confidence | 0.0728 | 0.2350 |
| Confidence >= 0.3: coverage, within 2 s | 0.0682, 0.8145 | 0.4156, 0.9267 |
| Confidence >= 0.5: coverage, within 2 s | 0.0058, 0.8857 | 0.1842, 0.9673 |

No tau was chosen.

Cost: GPU time for the classifier alone (RTX 5060 Laptop, 6 passes = 3 fold models x orig + hflip) is 4.50-4.54 s
per clip with windows, or 18.49-18.65 ms per window pass.

Commands:

    .venv-ml\Scripts\python.exe build_frame_cache.py --real --report-dir results\real_test\partA_cache --workers 8
    .venv-ml\Scripts\python.exe aggregate_runs.py --real-lock
    results\real_test\run_stages.ps1    # per run: analyse_predictions.py --run R --score-windows real --folds 0,1,2
                                        #   --out-dir results\real_test\scores\R; per stage: aggregate_runs.py --real-eval C1|C2

Details are in results/real_test/:
- RUN_ONCE.txt, PARTS_A_B.txt
- partA_cache/, partB_check/
- logs/, logs_chain.txt
- scores/<run>/fold_<k>/, predictions/<run>/
- stage_C1/report.txt and its CSVs
- stage_C2/report.txt, per_run_table.csv, final_table.csv, transfer_gap.csv, confidence_*_real.csv, gpu_seconds.csv,
  secondary_single_fold_models.csv

### Analysis of the saved predictions (addition to the plan, 2026-11-15)

This analysis was requested after both stages and was not in the plan. It ran no model and used only the CPU. Nothing
was chosen or changed, and RUN_ONCE.txt was only read. The command is `aggregate_runs.py --real-analysis` (new mode,
uncommitted, sha256 95605c4b...; the real test itself ran on commit cf2322d). Before computing anything new, the
script checks the saved predictions against the lock and reproduces every saved stage table it uses.

(1) Paired bootstrap of the ablation. It resamples source videos (clip stem without _NN) 10,000 times with numpy
seed 0 and reports percentile 95% intervals. There are 1944 source videos for the 2027 clips (at most 6 clips each)
and 400 for the frozen 400. Differences are C_v2_seed0 minus C_cctv:

| Comparison | Scope | T@1 [95% CI] | T@0.5 lower bound | T@2 lower bound | Within 2 s lower bound |
|---|---|---|---|---|---|
| vs C_cctv_seed0 | All 2027 | +0.0270 [0.0179, 0.0360] | above 0 | above 0 | above 0 |
| vs C_cctv mean over seeds | All 2027 | +0.0311 [0.0224, 0.0395] | above 0 | above 0 | above 0 |
| vs C_cctv_seed0 | Frozen 400 | +0.0229 [0.0016, 0.0442] | 0.0079 | -0.0073 | -0.0100 |
| vs C_cctv mean over seeds | Frozen 400 | +0.0190 [0.0000, 0.0380] | -0.0001 | -0.0017 | 0.0000 |

The bootstrap covers clip sampling only. It does not cover training-seed variation, and C_v2 has a single seed.

(2) Per-fold table: the synthetic validation T@1 of fold model k next to the real T@1 of that fold model alone. The
fold-2 models have the highest synthetic T@1 (0.858-0.878 against 0.594-0.624 for folds 0 and 1) and the highest real
T@1 on all 2027 clips in all four runs, but their real lead is small (0.379-0.422 against 0.353-0.411).

(3) Early impact (t* < 2.875 s). These clips are shorter: median duration 14.59 s for the 460 early clips against
29.88 s for the other 1567. The naive midpoint is higher on them (T@1 0.2764 vs 0.1641). The classifier's T@1 gain
over naive is similar in both groups:

| Predictor | Early clips | Other clips |
|---|---|---|
| C_v2 | +0.2359 | +0.2347 |
| C_cctv | +0.1966 +- 0.0072 | +0.2060 +- 0.0078 |

(4) Confidence, synthetic vs real, for each C_cctv seed (each clip counted once per table) and their mean over seeds.
The Wilson interval appears only as a secondary last column of the per-seed tables. Mean over seeds at
confidence >= 0.3:

| | Real | Synthetic |
|---|---|---|
| Coverage | 0.0682 | 0.4156 |
| Within 2 s | 0.8110 | 0.9266 |
| T@1 | 0.6117 | 0.8758 |

Details are in results/real_test/analysis/: report.txt, command.txt, run.log, bootstrap_ablation.csv,
source_videos.csv, per_fold_table.csv, early_impact.csv, confidence_per_seed.csv and
confidence_mean_over_seeds.csv.

## Zero-shot VLM full scan and tau (run once, 2026-11-16)

Synthetic clips only; no real clip or real label opened. Protocol frozen before the run in
results/vlm_full_scan/FROZEN.txt (2026-11-16T13:05:53Z, git commit 644f4e3 on branch vlm-zero-shot, sha256 of
vlm_zero_shot.py f5675fc8..., of the full-scan prompt v3 template 181e55a3... and of the narrow prompt v1 f0feb553...).
Model Qwen/Qwen3-VL-4B-Instruct, 4-bit NF4 (double quant, bf16 compute), greedy, max_new_tokens 64.
Full scan = version 3 (selected by the pre-registered rule on the 32 development clips): 16 distinct frames per pass,
each frame duplicated (one frame per temporal token group, own timestamp), 448 x 256 pixel area; coarse pass over the
whole clip -> t_c; fine pass over t_c +- 4 s -> t_fine; rules A (t < 0.3 s invalid) and B (fine answer at the first or
last frame -> t_c); coarse invalid -> duration / 2, fine invalid -> t_c. Narrow call = narrow v1 (kept by the
pre-registered rule): 16 frames over t_hat +- 2 s at 448 x 256.

Run: `.venv-vlm/Scripts/python.exe -u vlm_zero_shot.py --full-scan run` (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1),
2026-11-16 13:06:04Z to 17:23:10Z, one pass with no resume, on AC mains at start and at end (BatteryStatus 2, checked
every 25 clips). All 2211 clips; the 32 development clips gave the same raw answers as the v3 development run.
Report: `vlm_zero_shot.py --full-scan report`.

| p_full (within 2 s) | n | p_full | Wilson 95% (secondary) |
|---|---|---|---|
| all (pooled) | 2211 | 0.2741 (606/2211) | 0.2559-0.2931 |
| without the 32 dev clips | 2179 | 0.2740 (597/2179) | 0.2557-0.2931 |
| fold 0 | 770 | 0.2442 | 0.2151-0.2757 |
| fold 1 | 755 | 0.2596 | 0.2296-0.2920 |
| fold 2 | 686 | 0.3236 | 0.2897-0.3595 |
| head-on | 588 | 0.3197 | 0.2833-0.3585 |
| rear-end | 794 | 0.2305 | 0.2025-0.2610 |
| sideswipe | 405 | 0.2444 | 0.2051-0.2886 |
| single | 66 | 0.3182 | 0.2185-0.4379 |
| t-bone | 358 | 0.3212 | 0.2750-0.3713 |
| day | 1762 | 0.2804 | 0.2599-0.3018 |
| night | 449 | 0.2494 | 0.2117-0.2915 |

Full scan, all 2211: T@0.5 / T@1 / T@2 0.0907 / 0.1751 / 0.3436; coarse only within 2 s 0.2705; coarse fallbacks 219
(all rule A, i.e. coarse answers at 0.0: 219/2211 = 0.0991), fine fallbacks 347, gated by rule B 16, digit strings 0;
median |error| 3.650 s. 4.74 GPU s per clip (coarse + fine) on the NVIDIA GeForce RTX 5060 Laptop GPU; 6.96 wall s per
clip including CPU decoding.

Reference on the same 2211 clips (within 2 s): naive midpoint 0.3333 (T@1 0.2104); classifier C_cctv frozen
out-of-fold t_hat seeds 0 / 1 / 2 0.7544 / 0.7499 / 0.7494, mean 0.7512 +- 0.0028.

tau (frozen rule, computed once): the smallest c on results/diagnostics/tau_evidence/isotonic_curve.csv with
p_narrow(c) >= p_full = 0.2741 is **tau = 0.01** (p_narrow(0.01) = 0.3758; p_narrow(0.00) = 0.1818). Synthetic clips
with confidence >= tau (C_cctv frozen, out-of-fold): seeds 0 / 1 / 2 0.9579 / 0.9385 / 0.9634; pooled 0.9533
(6323/6633).

Details in results/vlm_full_scan/: FROZEN.txt, command.txt, run.log, run_events.txt, full_scan.csv,
raw_outputs.jsonl, report.txt, report_run.log.
