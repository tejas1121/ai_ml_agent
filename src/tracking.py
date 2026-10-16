"""Local MLflow tracking helpers."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log_run(
    dataset: str,
    method: str,
    seed: int,
    metrics: dict[str, float],
    params: dict[str, Any],
) -> None:
    """Log one result to the local SQLite store."""
    db_path = (PROJECT_ROOT / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("tabagent")
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "dataset": dataset,
                "method": method,
                "seed": str(seed),
            }
        )
        finite_metrics = {
            name: float(value)
            for name, value in metrics.items()
            if value is not None and math.isfinite(float(value))
        }
        mlflow.log_metrics(finite_metrics)
        mlflow.log_params(
            {
                name: value
                if isinstance(value, (str, bool, int, float))
                else str(value)
                for name, value in params.items()
                if value is not None
            }
        )


def _configure_tracking() -> None:
    db_path = (PROJECT_ROOT / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("tabagent")


@contextmanager
def agent_run(dataset: str, method: str, seed: int) -> Iterator[str]:
    """Keep one MLflow run open while the plain agent logs iteration metrics."""
    _configure_tracking()
    with mlflow.start_run(
        tags={
            "dataset": dataset,
            "method": method,
            "seed": str(seed),
        }
    ) as active_run:
        yield active_run.info.run_id


def log_agent_iteration(
    run_id: str,
    step: int,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
) -> None:
    """Log metrics and parameters to the active agent run at an iteration step."""
    active_run = mlflow.active_run()
    if active_run is None or active_run.info.run_id != run_id:
        raise RuntimeError("Agent iteration metrics require the matching active MLflow run.")
    finite_metrics = {
        name: float(value)
        for name, value in metrics.items()
        if value is not None and math.isfinite(float(value))
    }
    if finite_metrics:
        mlflow.log_metrics(finite_metrics, step=step)
    if params:
        mlflow.log_params(
            {
                name: value
                if isinstance(value, (str, bool, int, float))
                else str(value)
                for name, value in params.items()
                if value is not None
            }
        )
