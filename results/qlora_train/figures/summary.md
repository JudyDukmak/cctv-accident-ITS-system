# QLoRA evaluation summary (fold 1, window A, 755 synthetic windows)

| model | n | valid_answers | type_accuracy | S | mean_acc_S | mean_point_error | median_point_error | acc_day | S_day | acc_night | S_night | collision_visible_true | visible_agrees_t_star_inside | gpu_s_per_window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | 755 | 755 | 0.3457 | 0.2800 | 0.3128 | 0.2477 | 0.2291 | 0.3372 | 0.2724 | 0.3782 | 0.3091 | 0 | 0.3444 | 2.7313 |
| epoch 1 | 755 | 755 | 0.3060 | 0.6346 | 0.4703 | 0.1461 | 0.0857 | 0.3189 | 0.6412 | 0.2564 | 0.6089 | 755 | 0.6556 | 4.6700 |
| epoch 2 | 755 | 755 | 0.2318 | 0.6755 | 0.4537 | 0.1345 | 0.0565 | 0.2321 | 0.6855 | 0.2308 | 0.6372 | 740 | 0.6649 | 4.2249 |
| majority type (rear-end) | 755 | — | 0.3669 | — | — | — | — | 0.3656 | — | 0.3718 | — | — | — | — |
| image centre | 755 | — | — | 0.3851 | — | 0.2021 | 0.1799 | — | 0.3847 | — | 0.3865 | — | — | — |
