# Tabular agent experiment analysis

Inputs: `results/agent_results.csv`, `results/baselines.csv`, and live traces under `results/traces/` (384 trace records). `results/zoo_results.csv` is included when present. Live zoo candidate traces are read from `results/zoo_traces/`. All generated files are in `results/analysis/`.

## Mean test AUC by dataset and method

| dataset | xgboost_default | autogluon | plain | memory | zoo |
| --- | --- | --- | --- | --- | --- |
| blood-transfusion-service-center | 0.7015 | 0.7239 | 0.7450 | 0.7425 | 0.7515 |
| credit-approval | 0.9341 | 0.9482 | 0.9453 | 0.9437 | 0.9521 |
| credit-g | 0.7928 | 0.8010 | 0.7935 | 0.7935 | 0.7942 |
| diabetes | 0.7898 | 0.8204 | 0.8130 | 0.8281 | 0.8307 |
| ilpd | 0.7232 | 0.7646 | 0.7799 | 0.7599 | 0.7678 |
| kc1 | 0.8113 | 0.8339 | 0.8378 | 0.8361 | 0.8348 |
| phoneme | 0.9558 | 0.9640 | 0.9568 | 0.9590 | 0.9695 |
| wdbc | 0.9920 | 0.9928 | 0.9872 | 0.9884 | 0.9905 |

## Mean rank across datasets (1 = best)

| method | mean_rank |
| --- | --- |
| xgboost_default | 4.6250 |
| autogluon | 2.5000 |
| plain | 2.9375 |
| memory | 3.1875 |
| zoo | 1.7500 |

## Paired test-AUC comparisons

Difference is agent minus comparator. Dataset-level confidence intervals use 10,000 bootstrap resamples of datasets with fixed seed 20261020. The paired-run Wilcoxon result is explicitly labeled non-independent because seeds from one dataset are not independent datasets. Complete plain/memory pairs available: 24 of 24 plain runs.

| agent | comparator | paired_dataset_seed_n | dataset_n | mean_difference_paired_dataset_seed | mean_difference_dataset_means | dataset_mean_bootstrap_95_ci | wilcoxon_dataset_statistic | wilcoxon_dataset_p | wilcoxon_pairs_statistic_non_independent | wilcoxon_pairs_p_non_independent | dataset_wins | dataset_ties | dataset_losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memory | plain | 24 | 8 | -0.0009 | -0.0009 | [-0.007678, 0.005133] | 11.0000 | 0.6875 | 79.0000 | 0.519657 | 3 | 1 | 4 |
| plain | autogluon | 24 | 8 | 0.0012 | 0.0012 | [-0.005498, 0.008963] | 17.0000 | 0.945312 | 130.0000 | 0.583806 | 3 | 0 | 5 |
| plain | xgboost_default | 24 | 8 | 0.0197 | 0.0197 | [0.006133, 0.034692] | 3.0000 | 0.0390625 | 29.0000 | 0.00154869 | 7 | 0 | 1 |
| memory | autogluon | 24 | 8 | 0.0003 | 0.0003 | [-0.004748, 0.006577] | 16.0000 | 0.84375 | 147.0000 | 0.944107 | 3 | 0 | 5 |
| memory | xgboost_default | 24 | 8 | 0.0188 | 0.0188 | [0.006860, 0.030673] | 3.0000 | 0.0390625 | 29.0000 | 0.00264242 | 7 | 0 | 1 |
| zoo | xgboost_default | 24 | 8 | 0.0238 | 0.0238 | [0.011091, 0.036489] | 2.0000 | 0.0234375 | 21.0000 | 5.32866e-05 | 7 | 0 | 1 |
| zoo | autogluon | 24 | 8 | 0.0053 | 0.0053 | [-0.000536, 0.012630] | 8.0000 | 0.195312 | 104.0000 | 0.197378 | 6 | 0 | 2 |
| plain | zoo | 24 | 8 | -0.0041 | -0.0041 | [-0.009891, 0.002056] | 8.0000 | 0.195312 | 51.0000 | 0.0438037 | 2 | 0 | 6 |
| memory | zoo | 24 | 8 | -0.0050 | -0.0050 | [-0.007839, -0.002109] | 2.0000 | 0.0234375 | 49.0000 | 0.0208117 | 1 | 0 | 7 |

### Plain vs memory test AUC by dataset

Differences are memory minus plain; per-seed columns are individual paired differences.

| dataset | plain_mean | memory_mean | difference_seed_0 | difference_seed_1 | difference_seed_2 | difference_mean |
| --- | --- | --- | --- | --- | --- | --- |
| blood-transfusion-service-center | 0.7450 | 0.7425 | 0.0007 | -0.0156 | 0.0073 | -0.0025 |
| credit-approval | 0.9453 | 0.9437 | 0.0098 | 0.0045 | -0.0189 | -0.0016 |
| credit-g | 0.7935 | 0.7935 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| diabetes | 0.8130 | 0.8281 | 0.0078 | 0.0674 | -0.0300 | 0.0151 |
| ilpd | 0.7799 | 0.7599 | -0.0181 | -0.0074 | -0.0345 | -0.0200 |
| kc1 | 0.8378 | 0.8361 | 0.0062 | 0.0081 | -0.0195 | -0.0017 |
| phoneme | 0.9568 | 0.9590 | 0.0000 | 0.0000 | 0.0066 | 0.0022 |
| wdbc | 0.9872 | 0.9884 | -0.0094 | 0.0046 | 0.0083 | 0.0012 |

## Sensitivity: exclude incomplete or early-stopped agent pairs

Each listed run excludes its entire dataset-seed pair from all agent comparisons. Comparisons are recomputed using only complete dataset-seed pairs where both runs have `iterations_used >= 6` and `stop_reason == max_iterations`; baseline observations remain available for the eligible agent dataset-seeds. Excluded complete pairs: 1.

### Runs meeting an exclusion criterion

| dataset | seed | method | iterations_used | stop_reason | exclusion_reason |
| --- | --- | --- | --- | --- | --- |
| diabetes | 1 | plain | 1 | max_wall_seconds | iterations_used < 6; stop_reason != max_iterations |

### Sensitivity comparison table

| analysis | agent | comparator | paired_dataset_seed_n | dataset_n | mean_difference_paired_dataset_seed | mean_difference_dataset_means | dataset_mean_bootstrap_95_ci | wilcoxon_dataset_statistic | wilcoxon_dataset_p | wilcoxon_pairs_statistic_non_independent | wilcoxon_pairs_p_non_independent | dataset_wins | dataset_ties | dataset_losses | excluded_complete_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eligible complete runs only | memory | plain | 23 | 8 | -0.0039 | -0.0042 | [-0.009603, 0.000003] | 5.0000 | 0.15625 | 60.0000 | 0.266769 | 2 | 1 | 5 | 1 |
| eligible complete runs only | plain | autogluon | 23 | 8 | 0.0034 | 0.0038 | [-0.003318, 0.011336] | 13.0000 | 0.546875 | 130.0000 | 0.822927 | 4 | 0 | 4 | 1 |
| eligible complete runs only | plain | xgboost_default | 23 | 8 | 0.0206 | 0.0212 | [0.006969, 0.036353] | 3.0000 | 0.0390625 | 29.0000 | 0.00154869 | 7 | 0 | 1 | 1 |
| eligible complete runs only | memory | autogluon | 23 | 8 | -0.0005 | -0.0004 | [-0.004887, 0.005732] | 11.0000 | 0.382812 | 133.0000 | 0.893212 | 3 | 0 | 5 | 1 |
| eligible complete runs only | memory | xgboost_default | 23 | 8 | 0.0167 | 0.0170 | [0.006317, 0.028247] | 3.0000 | 0.0390625 | 29.0000 | 0.00454993 | 7 | 0 | 1 | 1 |
| eligible complete runs only | zoo | xgboost_default | 24 | 8 | 0.0238 | 0.0238 | [0.011091, 0.036489] | 2.0000 | 0.0234375 | 21.0000 | 5.32866e-05 | 7 | 0 | 1 | 1 |
| eligible complete runs only | zoo | autogluon | 24 | 8 | 0.0053 | 0.0053 | [-0.000536, 0.012630] | 8.0000 | 0.195312 | 104.0000 | 0.197378 | 6 | 0 | 2 | 1 |
| eligible complete runs only | plain | zoo | 23 | 8 | -0.0021 | -0.0022 | [-0.006607, 0.002997] | 10.0000 | 0.3125 | 51.0000 | 0.0766179 | 2 | 0 | 6 | 1 |
| eligible complete runs only | memory | zoo | 23 | 8 | -0.0060 | -0.0064 | [-0.009658, -0.002891] | 2.0000 | 0.0234375 | 30.0000 | 0.00511105 | 1 | 0 | 7 | 1 |

## Paired failures: errors and noops

Per-dataset values are means over paired seeds. Differences are memory minus plain. The dataset-level Wilcoxon uses dataset means; the seed-pair Wilcoxon is labeled non-independent.

| dataset | paired_seeds | error_count_plain_mean | error_count_memory_mean | error_count_difference_mean | noop_count_plain_mean | noop_count_memory_mean | noop_count_difference_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blood-transfusion-service-center | 3 | 0.3333 | 0.6667 | 0.3333 | 1.6667 | 1.3333 | -0.3333 |
| credit-approval | 3 | 3.3333 | 1.6667 | -1.6667 | 1.3333 | 1.0000 | -0.3333 |
| credit-g | 3 | 2.6667 | 0.6667 | -2.0000 | 1.3333 | 0.0000 | -1.3333 |
| diabetes | 3 | 1.6667 | 0.0000 | -1.6667 | 1.6667 | 0.3333 | -1.3333 |
| ilpd | 3 | 1.0000 | 0.0000 | -1.0000 | 1.3333 | 1.3333 | 0.0000 |
| kc1 | 3 | 2.0000 | 1.3333 | -0.6667 | 1.3333 | 0.6667 | -0.6667 |
| phoneme | 3 | 1.3333 | 0.0000 | -1.3333 | 0.6667 | 0.6667 | 0.0000 |
| wdbc | 3 | 2.3333 | 0.0000 | -2.3333 | 1.6667 | 0.0000 | -1.6667 |

| measure | paired_dataset_seed_n | dataset_n | mean_difference_paired_dataset_seed | mean_difference_dataset_means | wilcoxon_dataset_statistic | wilcoxon_dataset_p | wilcoxon_pairs_statistic_non_independent | wilcoxon_pairs_p_non_independent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| error_count | 24 | 8 | -1.2917 | -1.2917 | 1.0000 | 0.015625 | 22.5000 | 0.000873669 |
| noop_count | 24 | 8 | -0.7083 | -0.7083 | 0.0000 | 0.03125 | 45.5000 | 0.0117434 |

## Agent failure rates

`per_attempt_rate` divides category attempts by all LLM candidate attempts (the deterministic iteration-0 baseline is excluded). `runs_with_at_least_one_rate` is the fraction of agent runs with one or more attempts in that category. Mean failed iterations is reported per run.

| method | event | event_attempts | all_llm_attempts | per_attempt_rate | runs_with_at_least_one | agent_runs | runs_with_at_least_one_rate | mean_failed_iterations_per_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | error | 44 | 180 | 0.2444 | 22 | 24 | 0.9167 | 0.1667 |
| plain | invalid | 0 | 180 | 0.0000 | 0 | 24 | 0.0000 | 0.1667 |
| plain | timeout | 1 | 180 | 0.0056 | 1 | 24 | 0.0417 | 0.1667 |
| plain | noop | 33 | 180 | 0.1833 | 21 | 24 | 0.8750 | 0.1667 |
| memory | error | 13 | 156 | 0.0833 | 8 | 24 | 0.3333 | 0.0417 |
| memory | invalid | 0 | 156 | 0.0000 | 0 | 24 | 0.0000 | 0.0417 |
| memory | timeout | 0 | 156 | 0.0000 | 0 | 24 | 0.0000 | 0.0417 |
| memory | noop | 16 | 156 | 0.1026 | 8 | 24 | 0.3333 | 0.0417 |

## Cost by method

Baselines have zero prompt/completion tokens and LLM calls by definition; per-attempt LLM seconds are available only for agents. Mean and median compute seconds are adjacent columns.

| method | runs | compute_seconds_mean | compute_seconds_median | total_prompt_tokens | total_completion_tokens | mean_llm_calls | median_per_attempt_llm_seconds | attempts_over_600_seconds | over_600_seconds_flag | median_compute_ratio_vs_autogluon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_default | 24 | 1.2618 | 1.4136 | 0.0000 | 0.0000 | 0.0000 | n/a | 0 | NO | 0.0889 |
| autogluon | 24 | 28.6346 | 15.9099 | 0.0000 | 0.0000 | 0.0000 | n/a | 0 | NO | 1.0000 |
| plain | 24 | 296.5155 | 251.1376 | 178777.0000 | 25401.0000 | 7.5000 | 25.7741 | 1 | YES | 15.7850 |
| memory | 24 | 227.2689 | 210.9568 | 203779.0000 | 20666.0000 | 6.5000 | 25.4533 | 0 | NO | 13.2595 |
| zoo | 24 | 59.0247 | 57.6284 | 0.0000 | 0.0000 | 0.0000 | n/a | 0 | NO | 3.6222 |

## Selection optimism

Mean test AUC minus the selected run's best CV AUC. AutoGluon does not record a comparable best CV AUC in its baseline results.

| method | runs_with_both_auc_values | mean_test_auc_minus_best_cv_auc |
| --- | --- | --- |
| xgboost_default | 24 | 0.0200 |
| autogluon | 0 | n/a |
| plain | 24 | 0.0125 |
| memory | 24 | 0.0095 |
| zoo | 24 | 0.0092 |

### Individual attempts over 600 seconds

| method | dataset | seed | iteration | attempt | llm_seconds |
| --- | --- | --- | --- | --- | --- |
| plain | diabetes | 1 | 2 | 2 | 873.1127 |

## Per-attempt candidate quality

For each live plain/memory attempt at iterations 1+, quality is candidate CV AUC minus that run's iteration-0 CV AUC. The final share uses all attempts as its denominator; the preceding share uses only successful, non-noop attempts with a recorded baseline.

| method | all_llm_attempts | ok_non_noop_attempts_with_baseline | mean_cv_auc_delta_vs_iteration_0 | share_qualified_attempts_delta_gt_0_002 | share_all_attempts_ok_non_noop_delta_gt_0_002 |
| --- | --- | --- | --- | --- | --- |
| plain | 135 | 102 | 0.0056 | 0.5196 | 0.3926 |
| memory | 143 | 127 | 0.0029 | 0.5276 | 0.4685 |

## Classifier class at iteration 1

One first attempt per run is counted. Class names are extracted with the same tag parser used by experiment memory.

| method | classifier_class | runs | share_of_iteration_1_first_attempts |
| --- | --- | --- | --- |
| memory | LGBMClassifier | 6 | 0.2500 |
| memory | RandomForestClassifier | 17 | 0.7083 |
| memory | XGBClassifier | 1 | 0.0417 |
| plain | LightGBMClassifier | 4 | 0.1667 |
| plain | XGBClassifier | 20 | 0.8333 |
| zoo | LGBMClassifier | 24 | 1.0000 |

## CV learning curves

Only successful, non-noop candidates can improve the running best; failed/noop attempts carry the previous score forward. The clean subset contains 0 dataset-seed pairs with neither method having an error or noop at iterations 1-3. No pairs qualify, so clean-series values are unavailable.

| iteration | plain_all_successful_non_noop | memory_all_successful_non_noop | plain_clean_pairs | memory_clean_pairs | all_paired_dataset_seeds | clean_dataset_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8176 | 0.8176 | n/a | n/a | 24 | 0 |
| 1 | 0.8186 | 0.8342 | n/a | n/a | 24 | 0 |
| 2 | 0.8283 | 0.8414 | n/a | n/a | 24 | 0 |
| 3 | 0.8342 | 0.8425 | n/a | n/a | 24 | 0 |
| 4 | 0.8380 | 0.8458 | n/a | n/a | 24 | 0 |
| 5 | 0.8440 | 0.8460 | n/a | n/a | 24 | 0 |
| 6 | 0.8449 | 0.8469 | n/a | n/a | 24 | 0 |

### Paired memory-minus-plain CV differences

Dataset-level bootstrap confidence intervals use 10,000 dataset resamples.

| iteration | paired_dataset_seed_n | dataset_n | mean_difference_memory_minus_plain | dataset_mean_bootstrap_95_ci |
| --- | --- | --- | --- | --- |
| 1 | 24 | 8 | 0.0156 | [0.006003, 0.025644] |
| 2 | 24 | 8 | 0.0131 | [0.007045, 0.019081] |
| 3 | 24 | 8 | 0.0082 | [0.003225, 0.013525] |
| 6 | 24 | 8 | 0.0021 | [-0.000855, 0.004959] |
