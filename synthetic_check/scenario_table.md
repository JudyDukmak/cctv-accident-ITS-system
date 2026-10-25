# Synthetic scenarios

| scenario_id | map | crash_location | collision_type | n_views | n_clips | median_accident_time_s | current_val_fold | n_sampled | spread_m |
|---|---|---|---|---|---|---|---|---|---|
| Town03_head-on | Town03 | x=-84.7, y=-16.5 (n=13) | head-on | 52 | 212 | 3.65 | 2 | 13 | 8.9 |
| Town03_t-bone | Town03 | x=-8.7, y=-20.8 (n=18) | t-bone | 48 | 184 | 5.8 | 1 | 18 | 6.0 |
| Town04_rear-end | Town04 | x=15.9, y=-190.3 (n=8) | rear-end | 54 | 245 | 4.95 | 0 | 8 | 4.3 |
| Town04_sideswipe | Town04 | x=-11.6, y=52.7 (n=8) | sideswipe | 45 | 139 | 9.75 | 0 | 8 | 3.7 |
| Town05_head-on | Town05 | x=-188.6, y=9.2 (n=7) | head-on | 52 | 174 | 6.05 | 1 | 7 | 5.5 |
| Town05_rear-end | Town05 | x=-84.3, y=-92.5 (n=3) | rear-end | 87 | 277 | 10.05 | 1 | 3 | 16.0 |
| Town05_sideswipe | Town05 | x=-105.1, y=4.8 (n=9) | sideswipe | 46 | 107 | 7.3 | 1 | 9 | 5.6 |
| Town06_head-on | Town06 | x=569.1, y=54.9 (n=9) | head-on | 55 | 202 | 11.5 | 0 | 9 | 39.8 |
| Town06_sideswipe | Town06 | x=-182.9, y=148.0 (n=9) | sideswipe | 48 | 159 | 13.25 | 2 | 9 | 7.9 |
| Town07_rear-end | Town07 | x=-94.2, y=55.2 (n=8) | rear-end | 60 | 272 | 8.05 | 2 | 8 | 7.5 |
| Town07_single | Town07 | x=-98.3, y=-78.1 (n=3) | single | 11 | 13 | 12.8 | 1 | 3 | 3.0 |
| Town07_t-bone | Town07 | x=-3.4, y=-110.8 (n=3) | t-bone | 43 | 43 | 5.45 | 2 | 3 | 5.2 |
| Town10HD_single | Town10HD | x=-4.1, y=12.8 (n=3) | single | 31 | 53 | 4.85 | 0 | 3 | 4.1 |
| Town10HD_t-bone | Town10HD | x=-47.2, y=-54.2 (n=3) | t-bone | 49 | 131 | 6.0 | 0 | 3 | 4.1 |

# Maps

| map | n_scenarios | scenario_ids | n_clips | val_folds |
|---|---|---|---|---|
| Town03 | 2 | Town03_head-on, Town03_t-bone | 396 | {1, 2} |
| Town04 | 2 | Town04_rear-end, Town04_sideswipe | 384 | {0} |
| Town05 | 3 | Town05_head-on, Town05_rear-end, Town05_sideswipe | 558 | {1} |
| Town06 | 2 | Town06_head-on, Town06_sideswipe | 361 | {0, 2} |
| Town07 | 3 | Town07_rear-end, Town07_single, Town07_t-bone | 328 | {1, 2} |
| Town10HD | 2 | Town10HD_single, Town10HD_t-bone | 184 | {0} |
