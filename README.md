# TabAgent: Day 1 baselines

Local foundation, baselines, and experiment agents for **Does Experiment Memory
Help? A Budget-Matched Study of LLM Agents for Tabular AutoML**.

## Environment

Use the project virtual environment and install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

AutoGluon may not install on every Windows/Python combination. The batch runner
records a failed AutoGluon run and continues with the other runs; its error is
stored in the results CSV for the selected mode.

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
quick-run results are kept separately in `results/baselines_quick.csv`.
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

Print aggregate metrics and save `results/summary.csv` with:

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
