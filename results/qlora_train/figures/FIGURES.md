# QLoRA evaluation figures

All figures use only saved files: the validation answers of the untrained model (`baseline/`), epoch 1 and epoch 2 on **fold 1, window A (755 synthetic windows)**, `steps.csv`, the manifest (`results/qlora_data/manifest.csv`) and the cached frames. No model was run and no real clip or label was read. Every metric was recomputed from the per-window CSVs and checked against the saved metrics JSON (stop on any mismatch). Colours: untrained blue, epoch 1 orange, epoch 2 aqua in every figure; baselines grey dashed. Made by `make_figures.py` (this folder) with the `.venv-ml` interpreter.

Checkpoint choice (fixed rule, unchanged): epoch 1 (mean of accuracy and S 0.4703 vs 0.4537).

## 1. Confusion matrices — `fig1_confusion.png`
Counts (top) and row-normalised shares (bottom) for untrained, epoch 1 and epoch 2, with one colour scale per row. The untrained model answers mostly t-bone (535 of 755); after training the answers move to head-on: epoch 1 predicts head-on for 263 of 277 rear-end and 103 of 107 sideswipe windows, and epoch 2 predicts head-on in 584 of 755 windows. Single is never predicted correctly by any model (0, 0, 0 of 13).

## 2. Training curve — `fig2_training_curve.png`
Per-step loss on the answer tokens (light; batch 1, so one example per step) with a trailing 100-step moving average (dark), and the learning rate below. The 500-step means fall from steps 1-500 0.5422 to steps 6001-6308 0.3380; the dashed line marks the end of epoch 1 (step 3154) and the dotted line the resume point (step 4500), where the run paused on battery and was resumed from the saved checkpoint with the same code (sha256 identical in run.log). Steps 4224-4500 (277 steps) ran slower (GPU time > 6 s per step, likely on battery, which the power check only sees at a checkpoint); the slowdown changes speed, not the training data or settings.

All 500-step means: steps 1-500 0.5422; steps 501-1000 0.4927; steps 1001-1500 0.4803; steps 1501-2000 0.4596; steps 2001-2500 0.4442; steps 2501-3000 0.4268; steps 3001-3500 0.4120; steps 3501-4000 0.3833; steps 4001-4500 0.3808; steps 4501-5000 0.3615; steps 5001-5500 0.3469; steps 5501-6000 0.3387; steps 6001-6308 0.3380.

## 3. Location — `fig3_location_cdf.png`
Share of windows whose predicted box centre lies within x (normalised image units) of the true box centre. For small errors both trained epochs lie well above the untrained model and the image-centre baseline: the share with error below 0.1 is 0.1113 untrained, 0.5430 epoch 1, 0.5987 epoch 2 and 0.2013 for the image centre. The curves cross the image-centre curve at an error of about 0.350 (epoch 1) and 0.357 (epoch 2): the trained models have a heavier tail of far-off answers (error above 0.4 in 0.1073 and 0.1086 of windows, vs 0.0570 for the image centre and 0.1364 untrained). Median point error: 0.2291, 0.0857, 0.0565 and 0.1799.

## 4. Per-type accuracy and S — `fig4_per_type.png`
Left: type accuracy within each true type (recall); right: S within each true type. Horizontal lines are the overall baselines on all 755 windows (majority type rear-end 0.3669; image centre S 0.3851); the per-type baseline values are in `per_type.csv`. Training raises S for every type (epoch 1 highest for head-on 0.7428, lowest for single 0.5240), while type accuracy is high only for head-on and t-bone at epoch 1 and only for head-on at epoch 2.

## 5. Day vs night — `fig5_day_night.png`
Type accuracy and S for the 599 day and 156 night windows, with the overall baselines as lines. S is lower at night for both epochs (epoch 1 0.6412 day vs 0.6089 night; epoch 2 0.6855 vs 0.6372). The night group is small (n 156), so differences there are uncertain.

## 6. Examples — `fig6_examples.png`
Eight fold-1 window-A windows scored with the epoch-1 answers, showing the cached frame nearest t* with the true box (green) and the epoch-1 box (red). Selection rule, fixed before looking at the images: 2 each drawn at random (numpy default_rng(0), no repeats) from good location (point error < 0.05), bad location (> 0.30), correct type and wrong type; pool sizes and details are in `examples.csv`. In 5 of the 8, t* lies outside the 4-second window (marked OUTSIDE in the title), so the shown frame is the window edge, not the collision, and the true box refers to a moment the model did not see. These are illustrations, not a measurement.

## 7. Summary table — `summary.csv`, `summary.md`
Every metric for the three models and both baselines on the same 755 windows (— = not applicable). `per_type.csv` has the per-type numbers behind figure 4.

| model | n | valid_answers | type_accuracy | S | mean_acc_S | mean_point_error | median_point_error | acc_day | S_day | acc_night | S_night | collision_visible_true | visible_agrees_t_star_inside | gpu_s_per_window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained | 755 | 755 | 0.3457 | 0.2800 | 0.3128 | 0.2477 | 0.2291 | 0.3372 | 0.2724 | 0.3782 | 0.3091 | 0 | 0.3444 | 2.7313 |
| epoch 1 | 755 | 755 | 0.3060 | 0.6346 | 0.4703 | 0.1461 | 0.0857 | 0.3189 | 0.6412 | 0.2564 | 0.6089 | 755 | 0.6556 | 4.6700 |
| epoch 2 | 755 | 755 | 0.2318 | 0.6755 | 0.4537 | 0.1345 | 0.0565 | 0.2321 | 0.6855 | 0.2308 | 0.6372 | 740 | 0.6649 | 4.2249 |
| majority type (rear-end) | 755 | — | 0.3669 | — | — | — | — | 0.3656 | — | 0.3718 | — | — | — | — |
| image centre | 755 | — | — | 0.3851 | — | 0.2021 | 0.1799 | — | 0.3847 | — | 0.3865 | — | — | — |
