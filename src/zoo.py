"""Fixed no-LLM candidate zoo evaluated with the agent candidate sandbox."""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from src.agent import _default_code
from src.code_template import validate_code
from src.evaluate_final import final_test_auc
from src.sandbox import SandboxResult, run_candidate

PREPROCESSING = """\
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
"""


@dataclass(frozen=True)
class ZooCandidate:
    name: str
    code: str


@dataclass(frozen=True)
class CandidateResult:
    candidate_index: int
    candidate_name: str
    status: str
    cv_auc: float | None
    seconds: float
    error: str | None
    code: str


def _classifier_code(
    imports: str,
    constructor: str,
    *,
    scaler: bool = False,
) -> str:
    classifier_steps = (
        '            ("scaler", StandardScaler(with_mean=False)),\n'
        if scaler
        else ""
    )
    return (
        f"{imports}\n\n"
        "def build_model(X_train, y_train):\n"
        f"{PREPROCESSING}"
        "    return Pipeline(\n"
        "        steps=[\n"
        '            ("preprocessing", preprocessing),\n'
            f"{classifier_steps}"
        f'            ("classifier", {constructor}),\n'
        "        ]\n"
        "    )\n"
    )


def fixed_candidates(seed: int) -> list[ZooCandidate]:
    """Return the exact agent baseline followed by six fixed classifiers."""
    candidates = [
        ZooCandidate("xgboost_default", _default_code(seed)),
        ZooCandidate(
            "lightgbm",
            _classifier_code(
                "from lightgbm import LGBMClassifier",
                f"LGBMClassifier(random_state={seed}, n_jobs=6, verbosity=-1)",
            ),
        ),
        ZooCandidate(
            "catboost",
            _classifier_code(
                "from catboost import CatBoostClassifier",
                f"CatBoostClassifier(random_seed={seed}, verbose=False, "
                "thread_count=6, allow_writing_files=False)",
            ),
        ),
        ZooCandidate(
            "logistic_regression",
            _classifier_code(
                "from sklearn.linear_model import LogisticRegression",
                f"LogisticRegression(max_iter=1000, random_state={seed})",
                scaler=True,
            ),
        ),
        ZooCandidate(
            "random_forest",
            _classifier_code(
                "from sklearn.ensemble import RandomForestClassifier",
                f"RandomForestClassifier(random_state={seed}, n_jobs=6)",
            ),
        ),
        ZooCandidate(
            "gradient_boosting",
            _classifier_code(
                "from sklearn.ensemble import GradientBoostingClassifier",
                f"GradientBoostingClassifier(random_state={seed})",
            ),
        ),
        ZooCandidate(
            "extra_trees",
            _classifier_code(
                "from sklearn.ensemble import ExtraTreesClassifier",
                f"ExtraTreesClassifier(random_state={seed}, n_jobs=6)",
            ),
        ),
    ]
    for candidate in candidates:
        valid, reason = validate_code(candidate.code)
        if not valid:
            raise ValueError(
                f"Zoo candidate {candidate.name!r} failed validation: {reason}"
            )
    return candidates


def run_zoo_dataset(
    dataset: str,
    seed: int,
    on_candidate: Callable[[CandidateResult], None] | None = None,
) -> dict[str, Any]:
    """Evaluate train-only CV candidates and test the winner once."""
    candidates = fixed_candidates(seed)
    results: list[CandidateResult] = []
    started = time.perf_counter()

    for candidate_index, candidate in enumerate(candidates):
        result: SandboxResult = run_candidate(
            candidate.code,
            dataset,
            seed,
            timeout_s=120,
            mem_limit_mb=3000,
        )
        candidate_result = CandidateResult(
            candidate_index=candidate_index,
            candidate_name=candidate.name,
            status=result.status,
            cv_auc=result.cv_auc,
            seconds=result.seconds,
            error=result.error,
            code=candidate.code,
        )
        results.append(candidate_result)
        if on_candidate is not None:
            on_candidate(candidate_result)

    successful = [
        result
        for result in results
        if result.status == "ok" and result.cv_auc is not None
    ]
    if not successful:
        statuses = ", ".join(
            f"{result.candidate_name}={result.status}" for result in results
        )
        raise RuntimeError(
            f"No zoo candidate completed CV for {dataset} seed {seed}: {statuses}"
        )
    best = max(successful, key=lambda result: result.cv_auc)
    test_started = time.perf_counter()
    test_auc = final_test_auc(best.code, dataset, seed)
    total_seconds = time.perf_counter() - started
    return {
        "dataset": dataset,
        "method": "zoo",
        "seed": seed,
        "best_cv_auc": best.cv_auc,
        "test_auc": test_auc,
        "best_candidate": best.candidate_name,
        "n_candidates_ok": len(successful),
        "compute_seconds": total_seconds,
        "mode": "live",
        "test_seconds": time.perf_counter() - test_started,
        "candidate_results": results,
        "best_candidate_index": best.candidate_index,
    }
