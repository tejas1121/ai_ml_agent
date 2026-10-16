"""Subprocess execution of candidate estimators with time and memory limits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import psutil

from src.code_template import validate_code
from src.data import PROJECT_ROOT
from src.infra import is_infrastructure_error
from src.train_only import load_train_only

SandboxStatus = Literal[
    "ok", "timeout", "memory", "error", "invalid_code", "infra_error"
]
CANDIDATE_IMPORT_HEADER = """\
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
"""
RUNNER_SCRIPT = r'''
import importlib.util
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.data import get_cv_splits
from src.train_only import load_train_only

dataset, seed, candidate_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# Only training indices are selected; the held-out partition is never indexed or scored.
X_train, y_train = load_train_only(dataset, seed)

spec = importlib.util.spec_from_file_location("candidate", candidate_path)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the candidate module.")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

scores = []
for train_indices, validation_indices in get_cv_splits(X_train, y_train, seed).split(
    X_train, y_train
):
    X_fold_train = X_train.iloc[train_indices]
    y_fold_train = y_train.iloc[train_indices]
    X_validation = X_train.iloc[validation_indices]
    y_validation = y_train.iloc[validation_indices]
    model = candidate.build_model(X_fold_train, y_fold_train)
    model.fit(X_fold_train, y_fold_train)
    positive_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_validation)[:, positive_column]
    scores.append(float(roc_auc_score(y_validation, probabilities)))

print(json.dumps({"cv_auc": float(np.mean(scores))}))
'''


@dataclass(frozen=True)
class SandboxResult:
    status: SandboxStatus
    cv_auc: float | None
    error: str | None
    seconds: float
    peak_mem_mb: float


def _training_data(dataset: str, seed: int):
    """Load only the training partition for this seed."""
    return load_train_only(dataset, seed)


def _process_tree_memory_mb(pid: int) -> float:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0.0
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.NoSuchProcess:
        pass

    memory_bytes = 0
    for process in processes:
        try:
            memory_bytes += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return memory_bytes / (1024 * 1024)


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    root: psutil.Process | None = None
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []

    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if root is not None:
        try:
            root.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _last_lines(stderr: str, stdout: str) -> str:
    lines = (stderr or stdout).strip().splitlines()
    if not lines:
        return "Candidate runner exited without an error message."
    return "\n".join(lines[-15:])


def run_candidate(
    code: str,
    dataset: str,
    seed: int,
    timeout_s: float = 120,
    mem_limit_mb: float = 3000,
) -> SandboxResult:
    valid, reason = validate_code(code)
    if not valid:
        return SandboxResult("invalid_code", None, reason, 0.0, 0.0)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tabagent-candidate-") as temp_dir:
        temp_path = Path(temp_dir)
        candidate_path = temp_path / "candidate.py"
        runner_path = temp_path / "runner.py"
        candidate_path.write_text(
            f"{CANDIDATE_IMPORT_HEADER}\n\n{code}",
            encoding="utf-8",
        )
        runner_path.write_text(RUNNER_SCRIPT, encoding="utf-8")

        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "LOKY_MAX_CPU_COUNT",
        ):
            environment[name] = "6"
        environment["JOBLIB_MULTIPROCESSING"] = "0"
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (str(PROJECT_ROOT), existing_pythonpath)
            if item
        )

        process = subprocess.Popen(
            [
                sys.executable,
                str(runner_path),
                dataset,
                str(seed),
                str(candidate_path),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        status: SandboxStatus = "error"
        error: str | None = None
        peak_mem_mb = 0.0
        stop_monitoring = threading.Event()
        peak_memory_lock = threading.Lock()

        def monitor_memory() -> None:
            nonlocal peak_mem_mb, status, error
            while not stop_monitoring.wait(0.5):
                current_mem_mb = _process_tree_memory_mb(process.pid)
                with peak_memory_lock:
                    peak_mem_mb = max(peak_mem_mb, current_mem_mb)
                if current_mem_mb > mem_limit_mb:
                    status = "memory"
                    error = (
                        f"Process tree used {current_mem_mb:.1f} MB, exceeding "
                        f"the {mem_limit_mb:.1f} MB limit."
                    )
                    _kill_process_tree(process)
                    return

        monitor = threading.Thread(target=monitor_memory, daemon=True)
        monitor.start()
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            stdout_bytes, stderr_bytes = process.communicate()
            status = "timeout"
            error = f"Candidate exceeded the {timeout_s:g}-second timeout."
        finally:
            stop_monitoring.set()
            monitor.join()
        return_code = process.returncode
        seconds = time.perf_counter() - started
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

    if status in ("timeout", "memory"):
        return SandboxResult(status, None, error, seconds, peak_mem_mb)

    if return_code != 0:
        error = _last_lines(stderr, stdout)
        status: SandboxStatus = (
            "infra_error" if is_infrastructure_error(error) else "error"
        )
        return SandboxResult(
            status, None, error, seconds, peak_mem_mb
        )
    try:
        result_line = next(line for line in reversed(stdout.splitlines()) if line.strip())
        cv_auc = float(json.loads(result_line)["cv_auc"])
    except (StopIteration, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return SandboxResult(
            "error",
            None,
            _last_lines(stderr, stdout),
            seconds,
            peak_mem_mb,
        )
    return SandboxResult("ok", cv_auc, None, seconds, peak_mem_mb)
