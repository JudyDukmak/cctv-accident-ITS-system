# Classifier summary (synthetic cross-validation)

Best epoch per fold = last epoch marked best (val T@1). Pooled = best-epoch predictions of the three folds, 2211 clips, each exactly once. Validation only; no real data.

| run | config_file | config_hash | seed | level | best_epoch | best_check | n_clips | T_0.5 | T_1 | T_2 | within_2s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_v2_seed0 | videomae_v2_C.json | 1edaaedfeea7 | 0 | fold 0 | 0 | 2 | 770 | 0.4631 | 0.5828 | 0.6700 | 0.6584 |
| C_v2_seed0 | videomae_v2_C_rtx4050.json | fe4f55dc1c2f | 0 | fold 1 | 0 | 2 | 755 | 0.5227 | 0.6151 | 0.6912 | 0.6728 |
| C_v2_seed0 | videomae_v2_C.json | 1edaaedfeea7 | 0 | fold 2 | 1 | 6 | 686 | 0.7512 | 0.8518 | 0.8972 | 0.9023 |
| C_v2_seed0 | videomae_v2_C.json | 1edaaedfeea7 | 0 | pooled out-of-fold |  |  | 2211 | 0.5728 | 0.6773 | 0.7477 | 0.7390 |
| C_cctv_seed0 | videomae_v2_C_cctv.json | c890b4800516 | 0 | fold 0 | 0 | 2 | 770 | 0.4734 | 0.6032 | 0.6869 | 0.6857 |
| C_cctv_seed0 | videomae_v2_C_cctv_rtx4050.json | 31e3e00c8e12 | 0 | fold 1 | 0 | 3 | 755 | 0.4878 | 0.5986 | 0.6807 | 0.6623 |
| C_cctv_seed0 | videomae_v2_C_cctv.json | c890b4800516 | 0 | fold 2 | 1 | 6 | 686 | 0.7453 | 0.8564 | 0.9032 | 0.9096 |
| C_cctv_seed0 | videomae_v2_C_cctv.json | c890b4800516 | 0 | pooled out-of-fold |  |  | 2211 | 0.5627 | 0.6802 | 0.7519 | 0.7472 |
| C_cctv_seed1 | videomae_v2_C_cctv.json | c890b4800516 | 1 | fold 0 | 0 | 2 | 770 | 0.4571 | 0.5840 | 0.6676 | 0.6623 |
| C_cctv_seed1 | videomae_v2_C_cctv_rtx4050.json | 31e3e00c8e12 | 1 | fold 1 | 0 | 2 | 755 | 0.4846 | 0.6099 | 0.6980 | 0.6848 |
| C_cctv_seed1 | videomae_v2_C_cctv.json | c890b4800516 | 1 | fold 2 | 2 | 12 | 686 | 0.7394 | 0.8528 | 0.9028 | 0.9111 |
| C_cctv_seed1 | videomae_v2_C_cctv.json | c890b4800516 | 1 | pooled out-of-fold |  |  | 2211 | 0.5541 | 0.6762 | 0.7509 | 0.7472 |
| C_cctv_seed2 | videomae_v2_C_cctv.json | c890b4800516 | 2 | fold 0 | 0 | 2 | 770 | 0.4605 | 0.5869 | 0.6701 | 0.6675 |
| C_cctv_seed2 | videomae_v2_C_cctv_rtx4050.json | 31e3e00c8e12 | 2 | fold 1 | 0 | 2 | 755 | 0.5071 | 0.6121 | 0.6892 | 0.6715 |
| C_cctv_seed2 | videomae_v2_C_cctv.json | c890b4800516 | 2 | fold 2 | 2 | 12 | 686 | 0.7601 | 0.8699 | 0.9175 | 0.9257 |
| C_cctv_seed2 | videomae_v2_C_cctv.json | c890b4800516 | 2 | pooled out-of-fold |  |  | 2211 | 0.5694 | 0.6833 | 0.7534 | 0.7490 |
| C_cctv_seed0 + C_cctv_seed1 + C_cctv_seed2 | videomae_v2_C_cctv.json | c890b4800516 |  | seeds mean (n=3) |  |  |  | 0.5620 | 0.6799 | 0.7521 | 0.7478 |
| C_cctv_seed0 + C_cctv_seed1 + C_cctv_seed2 | videomae_v2_C_cctv.json | c890b4800516 |  | seeds sample std (ddof=1, n=3) |  |  |  | 0.0077 | 0.0036 | 0.0012 | 0.0010 |

## Across seeds

- videomae_v2_C_cctv.json (c890b4800516), n=3: T_0.5 0.5620 +- 0.0077; T_1 0.6799 +- 0.0036; T_2 0.7521 +- 0.0012; within_2s 0.7478 +- 0.0010

## Documented deviations

- C_v2_seed0: fold 1: config videomae_v2_C_rtx4050.json (hash fe4f55dc1c2f...) differs from fold 0 (videomae_v2_C.json, 1edaaedfeea7...) only in grad_accum 2 -> 4, micro_batch 8 -> 4; effective batch micro_batch x grad_accum = 16 in every fold
- C_v2_seed0: fold 1: documented deviation: cache rebuilt on another machine, core fields identical (run manifest results\diagnostics\C_v2_seed0_3folds\fold1_cache_manifest_rtx4050.json sha256 c76c46e6a09e... = recorded; current 4615b3475fb9...; differs only in created_at, skipped_this_run, written_this_run)
- C_v2_seed0: environment gpu (recorded, not checked): fold 0 NVIDIA GeForce RTX 5060 Laptop GPU, fold 1 NVIDIA GeForce RTX 4050 Laptop GPU, fold 2 NVIDIA GeForce RTX 5060 Laptop GPU
- C_v2_seed0: environment python (recorded, not checked): fold 0 3.12.2, fold 1 3.14.0, fold 2 3.12.2
- C_v2_seed0: environment git_commit (recorded, not checked): fold 0 91bd110, fold 1 e371121, fold 2 e371121
- C_cctv_seed0: fold 1: config videomae_v2_C_cctv_rtx4050.json (hash 31e3e00c8e12...) differs from fold 0 (videomae_v2_C_cctv.json, c890b4800516...) only in grad_accum 2 -> 4, micro_batch 8 -> 4; effective batch micro_batch x grad_accum = 16 in every fold
- C_cctv_seed0: fold 1: documented deviation: cache rebuilt on another machine, core fields identical (run manifest results\diagnostics\C_v2_seed0_3folds\fold1_cache_manifest_rtx4050.json sha256 c76c46e6a09e... = recorded; current 4615b3475fb9...; differs only in created_at, skipped_this_run, written_this_run)
- C_cctv_seed0: environment gpu (recorded, not checked): fold 0 NVIDIA GeForce RTX 5060 Laptop GPU, fold 1 NVIDIA GeForce RTX 4050 Laptop GPU, fold 2 NVIDIA GeForce RTX 5060 Laptop GPU
- C_cctv_seed0: environment python (recorded, not checked): fold 0 3.12.2, fold 1 3.14.0, fold 2 3.12.2
- C_cctv_seed1: fold 1: config videomae_v2_C_cctv_rtx4050.json (hash 31e3e00c8e12...) differs from fold 0 (videomae_v2_C_cctv.json, c890b4800516...) only in grad_accum 2 -> 4, micro_batch 8 -> 4; effective batch micro_batch x grad_accum = 16 in every fold
- C_cctv_seed1: fold 1: documented deviation: cache rebuilt on another machine, core fields identical (run manifest results\diagnostics\C_v2_seed0_3folds\fold1_cache_manifest_rtx4050.json sha256 c76c46e6a09e... = recorded; current 4615b3475fb9...; differs only in created_at, skipped_this_run, written_this_run)
- C_cctv_seed1: environment gpu (recorded, not checked): fold 0 NVIDIA GeForce RTX 5060 Laptop GPU, fold 1 NVIDIA GeForce RTX 4050 Laptop GPU, fold 2 NVIDIA GeForce RTX 5060 Laptop GPU
- C_cctv_seed1: environment python (recorded, not checked): fold 0 3.12.2, fold 1 3.14.0, fold 2 3.12.2
- C_cctv_seed1: environment git_commit (recorded, not checked): fold 0 80bac63, fold 1 a66ce3d, fold 2 80bac63
- C_cctv_seed2: fold 1: config videomae_v2_C_cctv_rtx4050.json (hash 31e3e00c8e12...) differs from fold 0 (videomae_v2_C_cctv.json, c890b4800516...) only in grad_accum 2 -> 4, micro_batch 8 -> 4; effective batch micro_batch x grad_accum = 16 in every fold
- C_cctv_seed2: fold 1: documented deviation: cache rebuilt on another machine, core fields identical (run manifest results\diagnostics\C_v2_seed0_3folds\fold1_cache_manifest_rtx4050.json sha256 c76c46e6a09e... = recorded; current 4615b3475fb9...; differs only in created_at, skipped_this_run, written_this_run)
- C_cctv_seed2: environment gpu (recorded, not checked): fold 0 NVIDIA GeForce RTX 5060 Laptop GPU, fold 1 NVIDIA GeForce RTX 4050 Laptop GPU, fold 2 NVIDIA GeForce RTX 5060 Laptop GPU
- C_cctv_seed2: environment python (recorded, not checked): fold 0 3.12.2, fold 1 3.14.0, fold 2 3.12.2
- C_cctv_seed2: environment git_commit (recorded, not checked): fold 0 80bac63, fold 1 a66ce3d, fold 2 80bac63
