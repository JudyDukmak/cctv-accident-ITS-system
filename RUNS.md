| std_seed0 | fold 0 | configs/videomae_v1.json (sha256 894d3436) | scheme B, standard aug | best epoch 0 | val T@1 0.4408 | within 2 s 0.488 | early stop after epoch 3 | diagnostics: results/diagnostics/std_seed0_fold0 |
| B_v2_seed0 | fold 0 only (no folds 1, 2) | configs/videomae_v2_B.json (sha256 5ee05e2f) | scheme B, standard aug, validation every 620 updates | best check 1 (epoch 1.00) | val T@1 0.4400 | within 2 s 0.487 | early stop after check 7 | diagnostics: results/diagnostics/B_v2_seed0_fold0 |
| C_v2_seed0 | folds 0, 1, 2 | folds 0, 2: configs/videomae_v2_C.json (sha256 1edaaedf); fold 1: videomae_v2_C_rtx4050.json (sha256 fe4f55dc), documented deviation | scheme C, standard aug, validation every 620 updates | best checks 2 / 2 / 6 (epoch 0.67 / 0.65 / 1.36) | val T@1 0.5828 / 0.6151 / 0.8518; pooled out-of-fold 0.6773 (2211 clips) | within 2 s 0.658 / 0.673 / 0.902; pooled 0.739 | early stop after checks 8 / 8 / 12 | results/classifier_summary.md; results/diagnostics/C_v2_seed0_3folds; results/diagnostics/C_v2_seed0_fold0, _fold1, _fold2 |
| C_cos3_seed0 | planned: folds 0, 1, 2 | folds 0, 2 (RTX 5060): configs/videomae_v2_C_cos3.json (sha256 56998505); fold 1 (RTX 4050): configs/videomae_v2_C_cos3_rtx4050.json (sha256 3fc8bd38) | scheme C, standard aug, cosine over 3 epochs, warmup 0.5 epoch, validation every 620 updates | - | - | - | - | not trained yet |

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
