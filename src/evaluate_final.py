"""One-time final evaluation, deliberately separate from candidate decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.code_template import validate_code
from src.data import load_dataset
from src.sandbox import CANDIDATE_IMPORT_HEADER


def final_test_auc(code: str, dataset: str, seed: int) -> float:
    """Score the selected best candidate once using the held-out test split.

    This function is for final reporting only. Its result must never be exposed
    to the LLM or to any candidate-selection/experiment loop.
    """
    valid, reason = validate_code(code)
    if not valid:
        raise ValueError(f"Candidate code failed validation: {reason}")

    namespace: dict[str, Any] = {}
    executable_code = f"{CANDIDATE_IMPORT_HEADER}\n\n{code}"
    exec(compile(executable_code, "<candidate>", "exec"), namespace)
    build_model = namespace["build_model"]

    X_train, X_test, y_train, y_test = load_dataset(dataset, seed)
    model = build_model(X_train, y_train)
    model.fit(X_train, y_train)
    positive_column = list(model.classes_).index(1)
    probabilities = np.asarray(model.predict_proba(X_test))[:, positive_column]
    return float(roc_auc_score(y_test, probabilities))
