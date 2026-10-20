# TabAgent: experiment code and frozen results

This repository contains the plain and memory LLM agents, a no-LLM zoo, and
conventional baselines for **Does Experiment Memory Help? A Budget-Matched
Study of LLM Agents for Tabular AutoML**.

## Results at a glance

Mean held-out test AUC over 8 datasets × 3 seeds (see
`results/analysis/summary.md` for every table, confidence interval and caveat):

| Method | Mean rank (1 = best) | Median compute per run |
| --- | --- | --- |
| zoo (fixed six-classifier sweep, no LLM) | 1.7500 | 57.6284 s |
| autogluon (medium quality, tree-only) | 2.5000 | 15.9099 s |
| plain LLM agent | 2.9375 | 251.1376 s |
| memory LLM agent | 3.1875 | 210.9568 s |
| xgboost_default | 4.6250 | 1.4136 s |

- Memory versus plain: mean test-AUC difference -0.0009 (95% bootstrap CI
  [-0.007678, 0.005133]); no benefit detected on final accuracy.
- Memory reduced code-error attempts from 0.2444 for plain to 0.0833 for
  memory. Memory's CV lead over plain was 0.0156 at iteration 1 and 0.0021
  at iteration 6.
- The no-LLM zoo matched or exceeded both agents (memory minus zoo -0.0050,
  95% CI [-0.007839, -0.002109]).
- All methods beat default XGBoost: mean paired differences were 0.0197
  (plain), 0.0188 (memory), 0.0238 (zoo), and 0.0053 (AutoGluon).
- All of these results come from 8 small datasets, 3 seeds and one quantized
  7B model, so treat them as descriptive. `results/analysis/summary.md`
  contains all tables and caveats.

Paper-writing handoff: see [HANDOFF.md](HANDOFF.md).

## Environment

Use the project virtual environment and install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

AutoGluon 1.6.3 installed and ran on Python 3.12 / Windows 11 for the frozen
results.

## Dataset profiles

From this directory, print profiles for all eight requested datasets (downloads
are cached below `data/openml`):

```powershell
.\.venv\Scripts\python.exe -m src.data
```

The profile output reports rows, columns, numeric/categorical feature counts,
missing-value rate, and class balance. Dataset loading encodes the binary
target as `0`/`1`, keeps categorical features as categories/objects, and makes
a stratified 80/20 split. The test split is only passed to the final test-score
calculation; cross-validation uses the training split only.

## Baselines

Smoke test (first two configured datasets, seed 0, AutoGluon limit 60 seconds):

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py --quick
```

Full batch (8 datasets × 3 seeds × 2 methods; AutoGluon limit 300 seconds):

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py
```

Each completed or failed full run is appended to `results/baselines.csv`;
quick-run results are written to `results/baselines_quick.csv`; the archived
smoke-test copy is in `results/dev_artifacts/baselines_quick.csv`.
Existing `(dataset, method, seed)` rows in the selected results CSV are skipped
on rerun. To retry a failed run, remove only its corresponding row from that
CSV before rerunning. Runs are also logged to the local SQLite-backed MLflow
store in `mlflow.db`. AutoGluon uses a temporary output directory which is
removed after each run. CPU parallelism is capped at six jobs.

AutoGluon baseline uses tree-based models only (LightGBM, CatBoost, XGBoost,
RandomForest, ExtraTrees, linear), no neural networks.

Start the MLflow UI from this directory:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The test AUC is final reporting only and is not used for fitting, cross-
validation, or model selection. XGBoost's CV AUC is calculated with shuffled,
stratified five-fold CV on the training split.

## Fixed no-LLM zoo baseline

Run the seven fixed candidates for the configured dataset/seed combinations:

```powershell
.\.venv\Scripts\python.exe scripts\run_zoo.py
```

The runner selects candidates only by the same five-fold, training-only CV
protocol used by the agents, then evaluates the winner on the held-out test
split once. Results are resumable in `results/zoo_results.csv`; candidate
traces are kept separately in `results/zoo_traces/` and are not used to build
the experiment-memory pool. For a smoke run, override the dataset, seed, and
output paths, for example:

```powershell
.\.venv\Scripts\python.exe scripts\run_zoo.py --datasets credit-g --seeds 0 `
  --results-file results/mock/zoo_smoke.csv `
  --traces-dir results/mock/zoo_traces
```

## Agent candidate structure

Both the plain and memory-agent proposal prompts use the same function
skeleton. Candidate code must derive numeric and categorical columns from
`X_train`, use a `ColumnTransformer` with imputation and
`OneHotEncoder(handle_unknown="ignore")` for categorical features, and return
a `Pipeline` with the estimator as its final step. Categorical columns arrive
as pandas category/object dtype. Keep encoders inside the pipeline. Never use
`LabelEncoder` or target encoders inside `build_model`: `LabelEncoder` is for
target columns only and breaks inside a `Pipeline`. For categorical features,
use `OneHotEncoder(handle_unknown="ignore")` or
`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`. Refer
to columns by name, never integer position. Target encoding or labeling `y` is
not needed. Candidates may use only the prompt's allowed imports.

## Experiment memory

Build the SQLite memory pool from explicitly marked live plain-agent traces:

```powershell
.\.venv\Scripts\python.exe scripts\build_memory.py
```

The builder skips iteration-zero baselines and infrastructure-error attempts.
It derives comparisons and compact model-change tags from training-only CV
results; it never uses final held-out scores. Mock traces and memory-agent
traces are not memory sources. Retrieval excludes all traces from the target
dataset, regardless of seed.

Run the memory method with the single-run interface:

```powershell
.\.venv\Scripts\python.exe scripts\run_agent.py --dataset credit-g --seed 0 --method memory
```

The default database is `data/memory.db`; override it with `--memory-db`.
Plain runs keep an empty memory context. Each trace records its mode, method,
and the exact context used in its header.

## Batch experiments

Run the plain phase, build the memory pool, and then run the memory phase:

```powershell
.\.venv\Scripts\python.exe scripts\run_batch.py --phase all
```

Use `--phase plain`, `--phase build`, or `--phase memory` to run an individual
phase. Batch runs are sequential and resumable from the selected agent-results
CSV. The build phase requires live plain traces for all 8 configured datasets
and 3 seeds unless `--allow-partial` is supplied. A mock dry run can use
separate outputs and the checked-in offline fixture traces:

```powershell
.\.venv\Scripts\python.exe scripts\run_batch.py --phase all --mock-llm `
  --datasets credit-g diabetes --seeds 0 --allow-partial `
  --results-file results/mock/batch.csv `
  --traces-dir results/mock/batch_traces `
  --memory-db results/mock/batch_memory.db
```

Generate the detailed report, tables and standard analysis figures under
`results/analysis/` with:

```powershell
.\.venv\Scripts\python.exe scripts\summarize.py
```

## Candidate execution harness

Before candidate code is executed in the sandbox or final evaluator, the
execution harness prepends a fixed import header for `numpy as np`,
`pandas as pd`, scikit-learn's `ColumnTransformer`, `make_column_selector`,
`SimpleImputer`, `Pipeline`, `make_pipeline`, `OneHotEncoder`, `OrdinalEncoder`,
`StandardScaler`, `MinMaxScaler`, and `RobustScaler`; common ensemble
classifiers (`AdaBoostClassifier`, `BaggingClassifier`, `ExtraTreesClassifier`,
`GradientBoostingClassifier`, `HistGradientBoostingClassifier`,
`RandomForestClassifier`, `StackingClassifier`, `VotingClassifier`); common
linear classifiers (`LogisticRegression`, `RidgeClassifier`, `SGDClassifier`,
`Perceptron`, `PassiveAggressiveClassifier`); and `XGBClassifier`,
`LGBMClassifier`, and `CatBoostClassifier`. The same header is applied
identically to every method. Static validation still checks only the model's
original code and continues to reject imports outside the allowed-module list.

## Frozen results and reproducibility

### Frozen record (2026-10-20)

These SHA-256 hashes identify the experiment inputs and frozen code record.
`results/analysis/` is derived from the frozen results and traces; it is not
frozen and can be regenerated with `scripts\summarize.py`.

| Artifact | SHA-256 |
| --- | --- |
| `results/agent_results.csv` | `172079542E4B64B7F7BE58334E3E118774F4BE3339F2700BCD8EAAFBE105B6FC` |
| `results/zoo_results.csv` | `2CF0CA3C88EBBE75C6B76DFF6192482224FA0C5E5AC9685B27678CCD92C0043F` |
| `results/baselines.csv` | `284C5C411C0D1804D92918FA81E52C46BA01BBFCF2B2ADE12C99AD83FE625F0E` |
| `data/memory.db` | `60E697B6881C66AC2CA8C1B044D591B148ABB495E4B67844BA5BC378F430EFCE` |
| `src/agent.py` | `54D7F4C930B87F53572570FA6A7B797CE80D2FD6A516463DC34015EA7D702ABC` |
| `src/code_template.py` | `9EF6D132BB23156DE116EE34D6EEC2D85444978C9A218CF8D9907139E83C2C3B` |
| `src/sandbox.py` | `36C0B68FFD94188E0F92CB0FADC68DB392AE46833316FD0FDB005C1397E100BB` |
| `src/memory.py` | `0F81ACF6B267DE90FB8F7C668E5EDBDB1C0AD2CBF487ECDCAD14BF252FE60863` |
| `src/evaluate_final.py` | `D379E8E36ACF6809645C7BA0941019FF623B1E23CC100407C62C0BC3BB1195D2` |
| `src/train_only.py` | `980FF6A9D51A254F443928F2A37165B3354C2E73FED0D5F1D3C37902476CF3EF` |

The results were produced at commit
`afdde397bc0add0e7224472b4f3997077a3c5cfc` (`zoo baseline and extended
analysis`); this identifies the results-producing commit, not a new commit.

### Setup

The frozen run environment used Python 3.12 and AutoGluon Tabular 1.6.3.
AutoGluon's configuration was `medium_quality`, with a 300-second time limit;
the configured model set was tree-based only, excluding `NN_TORCH` and
`FASTAI`. Although the per-run cap was 300 seconds, actual AutoGluon compute
time was about 16 seconds median. The full pinned environment is recorded in
`requirements.lock.txt`.

### Experiment design

- Datasets: the eight binary OpenML-CC18 datasets `credit-g`,
  `credit-approval`, `diabetes`, `blood-transfusion-service-center`, `ilpd`,
  `wdbc`, `kc1`, and `phoneme`.
- Seeds: `0`, `1`, and `2`. The seed also determines the stratified 80/20
  train/test split.
- Candidate selection uses five-fold cross-validation on the training split
  only. The held-out test split is used once per run for its final score, not
  for candidate selection.
- The agents use `qwen2.5-coder:7b` through local Ollama at temperature `0.2`,
  with seeded calls.
- Each agent run has a fixed iteration 0 followed by up to 6 proposal
  iterations, up to 2 repairs per iteration, and a 900-second compute budget.
- **plain** is the LLM agent without retrieved experiment memory.
- **memory** is the same agent with a read-only pool built only from plain
  traces. Retrieval is leave-one-dataset-out: traces from the target dataset
  are excluded.
- **zoo** is a no-LLM post-hoc control added after seeing the agent results.
  It evaluates the fixed iteration-0 default and a fixed six-classifier sweep
  (LightGBM, CatBoost, Logistic Regression, Random Forest, Gradient Boosting,
  and Extra Trees), selects by training-only CV AUC, and tests the winner.

### Execution harness notes

The agent's shared prompt and candidate skeleton are frozen in `src/agent.py`;
the shared code template is in `src/code_template.py`. Plain and memory runs
use the same shared prompt text; only the optional `memory_context` differs.
The candidate import header is shared by sandbox and final evaluation from
`src/sandbox.py`. No-op detection marks a candidate when its normalized code
repeats an earlier candidate or its CV AUC rounds to the incumbent's score at
six decimal places. The repair prompt provides the previous error and code,
asks for a targeted repair that preserves the same idea, and says not to
replace the candidate with the default XGBoost pipeline.

### Disclosures

- `wdbc` seed 2 and `ilpd` seed 2 (memory) were truncated by a system sleep.
  They were rerun with the same seeds after clearing the LLM cache; the
  original runs are retained in `backup_before_rerun/`.
- Plain `diabetes` seed 1 was affected by a stall: one LLM attempt took 873
  seconds. It was retained because plain traces form the memory pool. The
  sensitivity analysis in `results/analysis/summary.md` excludes it.
- `compute_seconds` uses the original LLM seconds stored in the cache; a
  repeated run reproduces the same text.
- The LLM cache (`data/llm_cache`) is git-ignored, so a fresh checkout
  regenerates responses. Exact replication of the agent runs is not guaranteed
  even with fixed seeds; the committed traces in `results/traces/` are the
  record of what was run.

### Reproduction commands

From the project directory, run these in order to reproduce the experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py
.\.venv\Scripts\python.exe scripts\run_batch.py --phase plain
.\.venv\Scripts\python.exe scripts\run_batch.py --phase build
.\.venv\Scripts\python.exe scripts\run_batch.py --phase memory
.\.venv\Scripts\python.exe scripts\run_zoo.py
.\.venv\Scripts\python.exe scripts\summarize.py
```

The project tests can be run with:

```powershell
.\.venv\Scripts\python.exe scripts\test_memory.py
.\.venv\Scripts\python.exe scripts\test_batch.py
```

Mock-only batch commands use `--mock-llm`; mock results, traces, and memory
databases are isolated under `results/mock/` and are never written to the live
experiment paths.

### Known limitations

- The study covers only eight small datasets and three seeds.
- The seed changes both the model randomness and the train/test split.
- The LLM results represent one quantized 7B model.
- Reported p-values are uncorrected and should be treated as descriptive.
- AutoGluon uses a `medium_quality`, tree-only configuration.
