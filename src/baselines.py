"""Non-LLM baseline estimators for the tabular benchmark."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data import get_cv_splits

logger = logging.getLogger(__name__)


def _filter_optional_torch_gpu_probe(record: logging.LogRecord) -> bool:
    return not record.getMessage().startswith(
        "\tFailed to import torch or check CUDA availability!"
    )


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = list(X.select_dtypes(include="number").columns)
    categorical_columns = list(X.select_dtypes(exclude="number").columns)

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def run_xgboost_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> tuple[float, float, float]:
    """Fit default-ish XGBoost; return test AUC, train-only CV AUC, elapsed seconds."""
    started = time.perf_counter()
    model = Pipeline(
        steps=[
            ("preprocessor", _preprocessor(X_train)),
            (
                "classifier",
                XGBClassifier(
                    random_state=seed,
                    n_jobs=6,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=get_cv_splits(X_train, y_train, seed),
        scoring="roc_auc",
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    positive_column = list(model.classes_).index(1)
    test_probabilities = model.predict_proba(X_test)[:, positive_column]
    test_auc = float(roc_auc_score(y_test, test_probabilities))
    return test_auc, float(np.mean(cv_scores)), time.perf_counter() - started


def run_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    time_limit: int = 300,
) -> tuple[float, float]:
    """Fit AutoGluon in an isolated temporary directory and return AUC and seconds.

    Failures are logged and re-raised so the runner can record the failed result
    and continue its batch without disguising an unsuccessful fit.
    """
    started = time.perf_counter()
    temporary_directory: str | None = None
    try:
        from autogluon.tabular import TabularPredictor

        gpu_probe_logger = logging.getLogger(
            "autogluon.common.utils.resource_utils"
        )
        if _filter_optional_torch_gpu_probe not in gpu_probe_logger.filters:
            gpu_probe_logger.addFilter(_filter_optional_torch_gpu_probe)

        with tempfile.TemporaryDirectory(prefix=f"tabagent-ag-{seed}-") as temp_dir:
            temporary_directory = temp_dir
            label = "__tabagent_target__"
            train_data = X_train.copy()
            train_data[label] = y_train.to_numpy()
            predictor = TabularPredictor(
                label=label,
                path=str(Path(temp_dir) / "predictor"),
                eval_metric="roc_auc",
                verbosity=0,
            )
            predictor.fit(
                train_data=train_data,
                presets="medium_quality",
                time_limit=time_limit,
                num_cpus=6,
                verbosity=0,
                excluded_model_types=["NN_TORCH", "FASTAI"],
            )
            probabilities = predictor.predict_proba(X_test)
            positive_column: Any = 1
            if positive_column not in probabilities.columns:
                raise ValueError(
                    "AutoGluon probability output did not include positive class 1."
                )
            test_auc = float(roc_auc_score(y_test, probabilities[positive_column]))
        return test_auc, time.perf_counter() - started
    except Exception:
        logger.exception(
            "AutoGluon failed (seed=%s, time_limit=%s, temporary_dir=%s)",
            seed,
            time_limit,
            temporary_directory,
        )
        raise
