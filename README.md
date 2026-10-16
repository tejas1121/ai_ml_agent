# TabAgent: Day 1 baselines

Local foundation and non-LLM baselines for **Does Experiment Memory Help? A
Budget-Matched Study of LLM Agents for Tabular AutoML**. This project does not
include LLM, LangGraph, or experiment-memory code.

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
