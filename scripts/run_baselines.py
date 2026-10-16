"""Resumable command-line runner for the non-LLM baseline batch."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import run_autogluon, run_xgboost_default
from src.data import load_dataset
from src.tracking import log_run

RESULT_COLUMNS = (
    "dataset",
    "method",
    "seed",
    "test_auc",
    "cv_auc",
    "seconds",
    "status",
    "error",
)
METHODS = ("xgboost_default", "autogluon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _read_finished_rows(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as result_file:
        reader = csv.DictReader(result_file)
        if reader.fieldnames != list(RESULT_COLUMNS):
            raise ValueError(
                f"Unexpected columns in {csv_path}: {reader.fieldnames!r}"
            )
        return {
            (row["dataset"], row["method"], int(row["seed"]))
            for row in reader
        }


def _append_result(csv_path: Path, result: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def _run_one(
    dataset: str,
    method: str,
    seed: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    autogluon_time_limit: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "test_auc": "",
        "cv_auc": "",
        "seconds": "",
        "status": "failed",
        "error": "",
    }
    started = time.perf_counter()
    try:
        if method == "xgboost_default":
            test_auc, cv_auc, seconds = run_xgboost_default(
                X_train, y_train, X_test, y_test, seed
            )
            result["cv_auc"] = cv_auc
        else:
            test_auc, seconds = run_autogluon(
                X_train,
                y_train,
                X_test,
                y_test,
                seed,
                time_limit=autogluon_time_limit,
            )
        result.update(test_auc=test_auc, seconds=seconds, status="completed")
        params: dict[str, Any] = {
            "status": "completed",
            "time_limit": autogluon_time_limit if method == "autogluon" else None,
            "n_jobs": 6,
        }
        metrics = {
            "test_auc": test_auc,
            "cv_auc": float(result["cv_auc"]) if result["cv_auc"] != "" else math.nan,
            "seconds": seconds,
        }
    except Exception as error:
        seconds = time.perf_counter() - started
        result["seconds"] = seconds
        result["error"] = f"{type(error).__name__}: {error}"
        logger.error(
            "Run failed for %s/%s/seed=%s:\n%s",
            dataset,
            method,
            seed,
            traceback.format_exc(),
        )
        params = {"status": "failed", "error": result["error"]}
        metrics = {"seconds": seconds}

    log_run(dataset, method, seed, metrics=metrics, params=params)
    return result


def _print_summary(csv_path: Path) -> None:
    if not csv_path.exists():
        print("No baseline results are available.")
        return
    results = pd.read_csv(csv_path)
    completed = results.loc[results["status"] == "completed"]
    if completed.empty:
        print("No completed baseline runs yet.")
        return
    summary = (
        completed.groupby(["dataset", "method"], as_index=False)["test_auc"]
        .mean()
        .pivot(index="dataset", columns="method", values="test_auc")
        .sort_index()
    )
    print("\nMean test ROC-AUC by dataset and method:")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the first two datasets, seed 0, and a 60-second AutoGluon limit.",
    )
    arguments = parser.parse_args()
    config = _load_config()
    datasets = config["datasets"][:2] if arguments.quick else config["datasets"]
    seeds = [0] if arguments.quick else config["seeds"]
    autogluon_time_limit = (
        config["time_limits"]["quick_autogluon"]
        if arguments.quick
        else config["time_limits"]["autogluon"]
    )
    csv_path = PROJECT_ROOT / config["paths"]["results"]
    if arguments.quick:
        csv_path = csv_path.with_name("baselines_quick.csv")
    finished = _read_finished_rows(csv_path)
    runs = [
        (dataset, method, seed)
        for dataset in datasets
        for seed in seeds
        for method in METHODS
        if (dataset, method, seed) not in finished
    ]

    for dataset, seed in tqdm(
        list(dict.fromkeys((dataset, seed) for dataset, _, seed in runs)),
        desc="Datasets/seeds",
    ):
        pending_methods = [
            method
            for method in METHODS
            if (dataset, method, seed) not in finished
        ]
        if not pending_methods:
            continue
        X_train, X_test, y_train, y_test = load_dataset(dataset, seed)
        for method in pending_methods:
            result = _run_one(
                dataset,
                method,
                seed,
                X_train,
                X_test,
                y_train,
                y_test,
                autogluon_time_limit,
            )
            _append_result(csv_path, result)
            finished.add((dataset, method, seed))
        del X_train, X_test, y_train, y_test

    _print_summary(csv_path)


if __name__ == "__main__":
    main()
