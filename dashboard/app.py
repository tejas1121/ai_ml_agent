"""Read-only dashboard for the frozen TabAgent experiment results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
METHODS = ("xgboost_default", "autogluon", "plain", "memory", "zoo")
AGENT_METHODS = ("plain", "memory")
RESULT_FILES = (
    "agent_results.csv",
    "baselines.csv",
    "zoo_results.csv",
)
ATTEMPT_COLUMNS = (
    "iteration",
    "attempt",
    "hypothesis",
    "status",
    "cv_auc",
    "noop",
    "error",
    "best_so_far",
)


def _read_csv_uncached(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


@st.cache_data(show_spinner=False)
def _cached_csv(relative_path: str, modified_ns: int, size: int) -> list[dict[str, str]]:
    del modified_ns, size
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen results file is missing: {path}")
    return _read_csv_uncached(path)


def read_csv(relative_path: str) -> list[dict[str, str]]:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen results file is missing: {path}")
    stat = path.stat()
    return _cached_csv(relative_path, stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def _cached_jsonl(path_text: str, modified_ns: int, size: int) -> list[dict[str, Any]]:
    del modified_ns, size
    path = Path(path_text)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in trace {path} at line {line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Trace record in {path} at line {line_number} is not an object."
                    )
                rows.append(value)
    return rows


@st.cache_data(show_spinner=False)
def read_traces() -> list[dict[str, Any]]:
    trace_dir = RESULTS_DIR / "traces"
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"Required live trace directory is missing: {trace_dir}")
    traces: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        stat = path.stat()
        records = _cached_jsonl(str(path), stat.st_mtime_ns, stat.st_size)
        if not records:
            continue
        header = records[0]
        if header.get("type") != "header" or header.get("mode") != "live":
            continue
        identity = _trace_identity(path, header)
        if identity is None:
            continue
        dataset, method, seed = identity
        for record in records[1:]:
            traces.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    **record,
                }
            )
    return traces


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _trace_identity(path: Path, header: dict[str, Any]) -> tuple[str, str, int] | None:
    parts = path.stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    dataset, filename_method, filename_seed = parts
    seed = _int(header.get("seed", filename_seed))
    method = str(header.get("method", filename_method))
    return (str(header.get("dataset", dataset)), method, seed) if seed is not None else None


def _completed_results() -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []

    for row in read_csv("results/baselines.csv"):
        if row.get("status") != "completed":
            continue
        method = row.get("method", "")
        if method not in ("xgboost_default", "autogluon"):
            continue
        combined.append(
            {
                "dataset": row.get("dataset", ""),
                "method": method,
                "seed": _int(row.get("seed")),
                "test_auc": _number(row.get("test_auc")),
                "compute_seconds": _number(row.get("seconds")),
            }
        )

    for row in read_csv("results/agent_results.csv"):
        if row.get("mode") != "live" or row.get("method") not in AGENT_METHODS:
            continue
        combined.append(
            {
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "seed": _int(row.get("seed")),
                "test_auc": _number(row.get("test_auc")),
                "compute_seconds": _number(row.get("compute_seconds")),
            }
        )

    for row in read_csv("results/zoo_results.csv"):
        if row.get("mode") != "live":
            continue
        combined.append(
            {
                "dataset": row.get("dataset", ""),
                "method": "zoo",
                "seed": _int(row.get("seed")),
                "test_auc": _number(row.get("test_auc")),
                "compute_seconds": _number(row.get("compute_seconds")),
            }
        )
    return [
        row
        for row in combined
        if row["dataset"]
        and row["test_auc"] is not None
        and row["compute_seconds"] is not None
    ]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rank_means(
    dataset_means: dict[str, dict[str, float]],
) -> dict[str, float]:
    ranks: dict[str, list[float]] = {method: [] for method in METHODS}
    for means in dataset_means.values():
        ordered = sorted(means.items(), key=lambda item: item[1], reverse=True)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            average_rank = ((index + 1) + end) / 2
            for method, _ in ordered[index:end]:
                ranks[method].append(average_rank)
            index = end
    return {
        method: float(_mean(method_ranks) or 0.0)
        for method, method_ranks in ranks.items()
        if method_ranks
    }


def _show_missing(error: Exception) -> None:
    st.error(str(error))
    st.stop()


def _render_overview() -> None:
    st.header("Overview")
    try:
        results = _completed_results()
    except (FileNotFoundError, ValueError, OSError) as error:
        _show_missing(error)
        return

    dataset_scores: dict[str, dict[str, list[float]]] = {}
    compute: dict[str, list[float]] = {method: [] for method in METHODS}
    for row in results:
        method = row["method"]
        dataset_scores.setdefault(row["dataset"], {}).setdefault(method, []).append(
            row["test_auc"]
        )
        compute[method].append(row["compute_seconds"])

    dataset_means: dict[str, dict[str, float]] = {
        dataset: {
            method: float(sum(scores) / len(scores))
            for method, scores in scores_by_method.items()
            if scores
        }
        for dataset, scores_by_method in dataset_scores.items()
    }
    dataset_table = [
        {
            "dataset": dataset,
            **{
                method: means.get(method)
                for method in METHODS
            },
        }
        for dataset, means in sorted(dataset_means.items())
    ]
    ranks = _rank_means(dataset_means)
    summary_table = [
        {
            "method": method,
            "mean_rank": ranks.get(method),
            "median_compute_seconds": median(compute[method]) if compute[method] else None,
        }
        for method in METHODS
    ]
    st.subheader("Mean test AUC by dataset and method")
    st.dataframe(dataset_table, hide_index=True, use_container_width=True)
    st.subheader("Mean rank and median compute")
    st.dataframe(summary_table, hide_index=True, use_container_width=True)


def _render_learning_curves() -> None:
    st.header("Learning curves")
    try:
        rows = read_csv("results/analysis/curves.csv")
    except (FileNotFoundError, ValueError, OSError) as error:
        _show_missing(error)
        return
    iterations = [_number(row.get("iteration")) for row in rows]
    plain = [_number(row.get("plain_all_successful_non_noop")) for row in rows]
    memory = [_number(row.get("memory_all_successful_non_noop")) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(iterations, plain, marker="o", label="plain")
    axis.plot(iterations, memory, marker="o", label="memory")
    axis.set(
        title="Best-so-far training CV AUC by iteration",
        xlabel="Iteration",
        ylabel="Mean best-so-far CV AUC",
        xticks=range(7),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_cost_accuracy() -> None:
    st.header("Cost vs accuracy")
    try:
        rows = read_csv("results/analysis/cost_vs_auc.csv")
    except (FileNotFoundError, ValueError, OSError) as error:
        _show_missing(error)
        return
    values = [
        row
        for row in rows
        if row.get("method") in METHODS
        and all(
            _number(row.get(column)) is not None
            for column in (
                "median_compute_seconds",
                "mean_test_auc",
                "bootstrap_95_ci_low",
                "bootstrap_95_ci_high",
            )
        )
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    offsets = {
        "xgboost_default": (8, 8),
        "autogluon": (8, 8),
        "plain": (8, 8),
        "memory": (8, -14),
        "zoo": (8, 8),
    }
    for row in values:
        x = float(_number(row["median_compute_seconds"]))
        y = float(_number(row["mean_test_auc"]))
        lower = y - float(_number(row["bootstrap_95_ci_low"]))
        upper = float(_number(row["bootstrap_95_ci_high"])) - y
        axis.errorbar(x, y, yerr=[[lower], [upper]], fmt="o", capsize=4)
        axis.annotate(row["method"], (x, y), xytext=offsets[row["method"]], textcoords="offset points")
    axis.set_xscale("log")
    axis.set(
        title="Test AUC versus median compute cost",
        xlabel="Median compute_seconds per run (seconds, log scale)",
        ylabel="Mean test AUC across dataset means",
    )
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_reliability() -> None:
    st.header("Reliability")
    try:
        traces = read_traces()
    except (FileNotFoundError, ValueError, OSError) as error:
        _show_missing(error)
        return
    attempts: dict[str, list[dict[str, Any]]] = {method: [] for method in AGENT_METHODS}
    for row in traces:
        if row.get("method") not in AGENT_METHODS or _int(row.get("iteration")) is None:
            continue
        if _int(row.get("iteration")) >= 1:
            attempts[row["method"]].append(row)

    summary: list[dict[str, Any]] = []
    for method in AGENT_METHODS:
        all_attempts = attempts[method]
        error_count = sum(row.get("status") == "error" for row in all_attempts)
        noop_count = sum(row.get("noop") is True for row in all_attempts)
        total = len(all_attempts)
        summary.append(
            {
                "method": method,
                "error_attempts": error_count,
                "all_attempts": total,
                "error_rate_per_attempt": error_count / total if total else None,
                "noop_attempts": noop_count,
                "noop_rate_per_attempt": noop_count / total if total else None,
            }
        )
    st.caption("Live traces only; iteration 0 is the deterministic baseline and is excluded.")
    st.dataframe(summary, hide_index=True, use_container_width=True)


def _agent_final_auc(
    dataset: str,
    method: str,
    seed: int,
) -> float | None:
    for row in read_csv("results/agent_results.csv"):
        if (
            row.get("mode") == "live"
            and row.get("dataset") == dataset
            and row.get("method") == method
            and _int(row.get("seed")) == seed
        ):
            return _number(row.get("test_auc"))
    return None


def _render_run_explorer() -> None:
    st.header("Run explorer")
    try:
        traces = read_traces()
        read_csv("results/agent_results.csv")
    except (FileNotFoundError, ValueError, OSError) as error:
        _show_missing(error)
        return
    available = [
        row
        for row in traces
        if row.get("method") in AGENT_METHODS and _int(row.get("iteration")) is not None
    ]
    run_keys = {
        (str(row["dataset"]), str(row["method"]), _int(row.get("seed")))
        for row in available
        if row.get("dataset") and _int(row.get("seed")) is not None
    }
    if not run_keys:
        st.info("No live plain or memory agent traces are available.")
        return

    datasets = sorted({dataset for dataset, _, _ in run_keys})
    dataset = st.selectbox("Dataset", datasets)
    methods = [
        method
        for method in AGENT_METHODS
        if any(item_dataset == dataset and item_method == method for item_dataset, item_method, _ in run_keys)
    ]
    if not methods:
        st.info(f"No plain or memory traces are available for {dataset}.")
        return
    method = st.selectbox("Method", methods)
    seeds = sorted(
        seed
        for item_dataset, item_method, seed in run_keys
        if item_dataset == dataset and item_method == method and seed is not None
    )
    seed = st.selectbox("Seed", seeds)
    header: dict[str, Any] | None = None
    run_attempts: list[dict[str, Any]] = []
    for path in sorted((RESULTS_DIR / "traces").glob("*.jsonl")):
        records = _cached_jsonl(str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if not records:
            continue
        candidate_header = records[0]
        identity = _trace_identity(path, candidate_header)
        if (
            candidate_header.get("type") == "header"
            and candidate_header.get("mode") == "live"
            and identity == (dataset, method, seed)
        ):
            header = candidate_header
            run_attempts = records[1:]
            break
    if header is None:
        st.error(f"No live trace found for {dataset}, {method}, seed {seed}.")
        return

    display_rows = [
        {column: row.get(column) for column in ATTEMPT_COLUMNS}
        for row in run_attempts
    ]
    st.subheader(f"{dataset} · {method} · seed {seed}")
    st.dataframe(display_rows, hide_index=True, use_container_width=True)

    labels = [
        f"iteration {row.get('iteration')} · attempt {row.get('attempt')}"
        for row in run_attempts
    ]
    coded_indices = [
        index for index, row in enumerate(run_attempts) if row.get("code")
    ]
    if coded_indices:
        chosen_label = st.selectbox(
            "Attempt code",
            [labels[index] for index in coded_indices],
        )
        chosen_index = coded_indices[labels.index(chosen_label)]
        with st.expander("Generated candidate code"):
            st.code(str(run_attempts[chosen_index]["code"]), language="python")
    if method == "memory":
        st.subheader("Memory context used")
        st.text(str(header.get("memory_context", "")))

    best_values = [
        _number(row.get("best_so_far"))
        for row in run_attempts
        if _number(row.get("best_so_far")) is not None
    ]
    if best_values:
        st.metric("Best-so-far CV AUC", f"{max(best_values):.6f}")

    test_auc = _agent_final_auc(dataset, method, seed)
    if test_auc is None:
        st.info("No final test AUC is available for this run.")
    else:
        st.metric("Final test AUC (reporting only)", f"{test_auc:.6f}")
        st.caption("Held-out test AUC is reporting only; it is never used for selection.")


st.set_page_config(page_title="Frozen TabAgent Results", layout="wide")
st.title("Frozen TabAgent Results")
st.info(
    "Frozen results; read-only view. Held-out test AUC is for reporting, never for selection."
)
st.caption(
    "P-values and the sensitivity analysis are in results/analysis/summary.md."
)
page = st.sidebar.radio(
    "Page",
    ("Overview", "Learning curves", "Cost vs accuracy", "Reliability", "Run explorer"),
)

if page == "Overview":
    _render_overview()
elif page == "Learning curves":
    _render_learning_curves()
elif page == "Cost vs accuracy":
    _render_cost_accuracy()
elif page == "Reliability":
    _render_reliability()
else:
    _render_run_explorer()
