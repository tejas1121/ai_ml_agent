# Tabular agent experiment analysis

Inputs: `results/agent_results.csv`, `results/baselines.csv`, and live traces under `results/traces/` (384 trace records). All generated files are in `results/analysis/`.

## Mean test AUC by dataset and method

| dataset | xgboost_default | autogluon | plain | memory |
| --- | --- | --- | --- | --- |
| blood-transfusion-service-center | 0.7015 | 0.7239 | 0.7450 | 0.7425 |
| credit-approval | 0.9341 | 0.9482 | 0.9453 | 0.9437 |
| credit-g | 0.7928 | 0.8010 | 0.7935 | 0.7935 |
| diabetes | 0.7898 | 0.8204 | 0.8130 | 0.8281 |
| ilpd | 0.7232 | 0.7646 | 0.7799 | 0.7599 |
| kc1 | 0.8113 | 0.8339 | 0.8378 | 0.8361 |
| phoneme | 0.9558 | 0.9640 | 0.9568 | 0.9590 |
| wdbc | 0.9920 | 0.9928 | 0.9872 | 0.9884 |

## Mean rank across datasets (1 = best)

| method | mean_rank |
| --- | --- |
| xgboost_default | 3.7500 |
| autogluon | 1.7500 |
| plain | 2.1875 |
| memory | 2.3125 |

## Paired test-AUC comparisons

Difference is agent minus comparator. Dataset-level confidence intervals use 10,000 bootstrap resamples of datasets with fixed seed 20261020. The paired-run Wilcoxon result is explicitly labeled non-independent because seeds from one dataset are not independent datasets. Complete plain/memory pairs available: 24 of 24 plain runs.

| agent | comparator | paired_dataset_seed_n | dataset_n | mean_difference_paired_dataset_seed | mean_difference_dataset_means | dataset_mean_bootstrap_95_ci | wilcoxon_dataset_statistic | wilcoxon_dataset_p | wilcoxon_pairs_statistic_non_independent | wilcoxon_pairs_p_non_independent | dataset_wins | dataset_ties | dataset_losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | autogluon | 24 | 8 | 0.0012 | 0.0012 | [-0.005498, 0.008963] | 17.0000 | 0.945312 | 130.0000 | 0.583806 | 3 | 0 | 5 |
| plain | xgboost_default | 24 | 8 | 0.0197 | 0.0197 | [0.006485, 0.034657] | 3.0000 | 0.0390625 | 29.0000 | 0.00154869 | 7 | 0 | 1 |
| memory | autogluon | 24 | 8 | 0.0003 | 0.0003 | [-0.004797, 0.006564] | 16.0000 | 0.84375 | 147.0000 | 0.944107 | 3 | 0 | 5 |
| memory | xgboost_default | 24 | 8 | 0.0188 | 0.0188 | [0.007125, 0.030577] | 3.0000 | 0.0390625 | 29.0000 | 0.00264242 | 7 | 0 | 1 |

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

Baselines have zero prompt/completion tokens and LLM calls by definition; per-attempt LLM seconds are available only for agents.

| method | runs | mean_compute_seconds | median_compute_seconds | mean_prompt_tokens | mean_completion_tokens | mean_llm_calls | median_per_attempt_llm_seconds | attempts_over_600_seconds | over_600_seconds_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_default | 24 | 1.2618 | 1.4136 | 0.0000 | 0.0000 | 0.0000 | n/a | 0 | NO |
| autogluon | 24 | 28.6346 | 15.9099 | 0.0000 | 0.0000 | 0.0000 | n/a | 0 | NO |
| plain | 24 | 296.5155 | 251.1376 | 7449.0417 | 1058.3750 | 7.5000 | 25.7741 | 1 | YES |
| memory | 24 | 227.2689 | 210.9568 | 8490.7917 | 861.0833 | 6.5000 | 25.4533 | 0 | NO |

### Individual attempts over 600 seconds

| method | dataset | seed | iteration | attempt | llm_seconds |
| --- | --- | --- | --- | --- | --- |
| plain | diabetes | 1 | 2 | 2 | 873.1127 |

## CV learning curves

Mean best-so-far training CV AUC for iterations 0-6. Failed or missing iterations carry the preceding best score forward. `curves.csv` and `curves.png` contain the plotted series.

| iteration | plain | memory |
| --- | --- | --- |
| 0 | 0.8176 | 0.8176 |
| 1 | 0.8186 | 0.8342 |
| 2 | 0.8283 | 0.8414 |
| 3 | 0.8342 | 0.8425 |
| 4 | 0.8380 | 0.8458 |
| 5 | 0.8440 | 0.8460 |
| 6 | 0.8449 | 0.8469 |
