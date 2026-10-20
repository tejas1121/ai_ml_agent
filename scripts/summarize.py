"""Write analysis tables and CV curves from saved baseline and agent results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import _change_tags

BASELINE_PATH = PROJECT_ROOT / "results" / "baselines.csv"
AGENT_PATH = PROJECT_ROOT / "results" / "agent_results.csv"
ZOO_PATH = PROJECT_ROOT / "results" / "zoo_results.csv"
TRACE_DIR = PROJECT_ROOT / "results" / "traces"
ZOO_TRACE_DIR = PROJECT_ROOT / "results" / "zoo_traces"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
METHODS = ("xgboost_default", "autogluon", "plain", "memory", "zoo")
AGENT_METHODS = ("plain", "memory")
BASELINE_METHODS = ("autogluon", "xgboost_default")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_020
MAX_ATTEMPT_SECONDS = 600.0


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baselines = _read_csv(BASELINE_PATH)
    agents = _read_csv(AGENT_PATH)
    zoo = _read_csv(ZOO_PATH) if ZOO_PATH.exists() else pd.DataFrame()
    if not baselines.empty:
        baselines = baselines.copy()
        _numeric(
            baselines,
            ("seed", "test_auc", "cv_auc", "seconds"),
        )
        baselines["method"] = baselines["method"].astype(str)
        baselines["compute_seconds"] = baselines.get("seconds", np.nan)
        baselines["best_cv_auc"] = baselines.get("cv_auc", np.nan)
        baselines["prompt_tokens"] = 0.0
        baselines["completion_tokens"] = 0.0
        baselines["llm_calls"] = 0.0
    if not agents.empty:
        agents = agents.copy()
        _numeric(
            agents,
            (
                "seed",
                "test_auc",
                "best_cv_auc",
                "compute_seconds",
                "prompt_tokens",
                "completion_tokens",
                "llm_calls",
                "n_failed_iterations",
                "iterations_used",
            ),
        )
        if "mode" in agents.columns:
            agents = agents.loc[agents["mode"].eq("live")].copy()
    if not zoo.empty:
        zoo = zoo.copy()
        _numeric(
            zoo,
            (
                "seed",
                "test_auc",
                "best_cv_auc",
                "compute_seconds",
                "n_candidates_ok",
            ),
        )
        zoo["method"] = "zoo"
        if "mode" in zoo.columns:
            zoo = zoo.loc[zoo["mode"].eq("live")].copy()

    baseline_columns = [
        "dataset",
        "method",
        "seed",
        "test_auc",
        "best_cv_auc",
        "compute_seconds",
        "prompt_tokens",
        "completion_tokens",
        "llm_calls",
    ]
    agent_columns = [
        "dataset",
        "method",
        "seed",
        "test_auc",
        "best_cv_auc",
        "compute_seconds",
        "prompt_tokens",
        "completion_tokens",
        "llm_calls",
    ]
    zoo_columns = [
        "dataset",
        "method",
        "seed",
        "test_auc",
        "best_cv_auc",
        "compute_seconds",
        "prompt_tokens",
        "completion_tokens",
        "llm_calls",
    ]
    if not zoo.empty:
        for column in ("prompt_tokens", "completion_tokens", "llm_calls"):
            zoo[column] = 0.0
    frames = [
        frame[columns]
        for frame, columns in (
            (baselines, baseline_columns),
            (agents, agent_columns),
            (zoo, zoo_columns),
        )
        if not frame.empty
    ]
    combined = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame(columns=baseline_columns)
    )
    return baselines, agents, combined


def _load_live_traces() -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    if not TRACE_DIR.exists():
        return traces

    for path in sorted(TRACE_DIR.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        if not records or not isinstance(records[0], dict):
            continue
        header = records[0]
        if header.get("type") != "header" or header.get("mode") != "live":
            continue
        name_parts = path.stem.rsplit("_", 2)
        if len(name_parts) != 3:
            continue
        dataset, method, seed_text = name_parts
        if method not in AGENT_METHODS:
            continue
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        for record in records[1:]:
            if isinstance(record, dict):
                traces.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        **record,
                    }
                )
    return traces


def _load_live_zoo_traces() -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    if not ZOO_TRACE_DIR.exists():
        return traces
    for path in sorted(ZOO_TRACE_DIR.glob("*.jsonl")):
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records or not isinstance(records[0], dict):
            continue
        header = records[0]
        if (
            header.get("type") != "header"
            or header.get("mode") != "live"
            or header.get("method") != "zoo"
        ):
            continue
        for record in records[1:]:
            if isinstance(record, dict):
                traces.append(
                    {
                        "dataset": header.get("dataset"),
                        "method": "zoo",
                        "seed": header.get("seed"),
                        **record,
                    }
                )
    return traces


def _per_dataset_auc(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=list(METHODS))
    methods = [method for method in METHODS if method in set(combined["method"])]
    return combined.pivot_table(
        index="dataset",
        columns="method",
        values="test_auc",
        aggfunc="mean",
    ).reindex(columns=methods)


def _available_baseline_methods(combined: pd.DataFrame) -> tuple[str, ...]:
    if "zoo" in set(combined.get("method", pd.Series(dtype=object))):
        return (*BASELINE_METHODS, "zoo")
    return BASELINE_METHODS


def _mean_ranks(per_dataset: pd.DataFrame) -> pd.Series:
    if per_dataset.empty:
        return pd.Series(index=per_dataset.columns, dtype=float)
    return per_dataset.rank(axis=1, ascending=False, method="average").mean()


def _wilcoxon_summary(differences: pd.Series) -> tuple[str, str]:
    values = differences.dropna().to_numpy(dtype=float)
    if len(values) < 5:
        return "n/a", f"not computed (only {len(values)} observations; need 5)"
    try:
        statistic, p_value = wilcoxon(values, alternative="two-sided")
    except ValueError as error:
        return "n/a", f"not computed ({error})"
    return f"{statistic:.4f}", f"{p_value:.6g}"


def _paired_comparison_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparison_pairs = [
        ("memory", "plain"),
        *(
            (agent_method, baseline_method)
            for agent_method in AGENT_METHODS
            for baseline_method in BASELINE_METHODS
        ),
    ]
    if "zoo" in set(combined.get("method", pd.Series(dtype=object))):
        comparison_pairs.extend(
            (
                ("zoo", "xgboost_default"),
                ("zoo", "autogluon"),
                ("plain", "zoo"),
                ("memory", "zoo"),
            )
        )
    for agent_method, comparator in comparison_pairs:
        pair_data = combined.loc[
            combined["method"].isin((agent_method, comparator)),
            ["dataset", "seed", "method", "test_auc"],
        ]
        if pair_data.empty:
            paired = pd.DataFrame(columns=[agent_method, comparator])
        else:
            paired = pair_data.pivot_table(
                index=["dataset", "seed"],
                columns="method",
                values="test_auc",
                aggfunc="mean",
            )
            paired = paired.reindex(columns=[agent_method, comparator])
            paired = paired.dropna(subset=[agent_method, comparator])
        rows.append(_paired_comparison_row(paired, agent_method, comparator))
    return pd.DataFrame(rows)


def _paired_comparison_row(
    paired: pd.DataFrame,
    agent_method: str,
    comparator: str,
) -> dict[str, Any]:
    differences = paired[agent_method] - paired[comparator]
    dataset_differences = differences.groupby(level="dataset").mean()
    if dataset_differences.empty:
        ci_low = ci_high = np.nan
    else:
        random = np.random.default_rng(BOOTSTRAP_SEED)
        samples = random.choice(
            dataset_differences.to_numpy(dtype=float),
            size=(BOOTSTRAP_RESAMPLES, len(dataset_differences)),
            replace=True,
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])

    dataset_stat, dataset_p = _wilcoxon_summary(dataset_differences)
    pair_stat, pair_p = _wilcoxon_summary(differences)
    return {
        "agent": agent_method,
        "comparator": comparator,
        "paired_dataset_seed_n": len(differences),
        "dataset_n": len(dataset_differences),
        "mean_difference_paired_dataset_seed": differences.mean(),
        "mean_difference_dataset_means": dataset_differences.mean(),
        "dataset_mean_bootstrap_95_ci": (
            f"[{ci_low:.6f}, {ci_high:.6f}]"
            if np.isfinite(ci_low) and np.isfinite(ci_high)
            else "n/a"
        ),
        "wilcoxon_dataset_statistic": dataset_stat,
        "wilcoxon_dataset_p": dataset_p,
        "wilcoxon_pairs_statistic_non_independent": pair_stat,
        "wilcoxon_pairs_p_non_independent": pair_p,
        "dataset_wins": int((dataset_differences > 0).sum()),
        "dataset_ties": int((dataset_differences == 0).sum()),
        "dataset_losses": int((dataset_differences < 0).sum()),
    }


def _full_iteration_sensitivity(
    agents: pd.DataFrame,
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Recompute comparisons after excluding pairs with incomplete agent runs."""
    required = {
        "dataset",
        "seed",
        "method",
        "test_auc",
        "iterations_used",
        "stop_reason",
    }
    if agents.empty or not required.issubset(agents.columns):
        return (
            _paired_comparison_table(
                combined.loc[
                    combined["method"].isin(_available_baseline_methods(combined))
                ]
            ),
            pd.DataFrame(
                columns=[
                    "dataset",
                    "seed",
                    "method",
                    "iterations_used",
                    "stop_reason",
                    "exclusion_reason",
                ]
            ),
            0,
        )
    selected = agents.loc[
        agents["method"].isin(AGENT_METHODS),
        [
            "dataset",
            "seed",
            "method",
            "test_auc",
            "iterations_used",
            "stop_reason",
        ],
    ].copy()
    selected["iterations_used"] = pd.to_numeric(
        selected["iterations_used"], errors="coerce"
    )
    selected["eligible"] = selected["iterations_used"].ge(6) & selected[
        "stop_reason"
    ].eq("max_iterations")

    failing = selected.loc[~selected["eligible"]].copy()
    failing["exclusion_reason"] = failing.apply(
        lambda row: "; ".join(
            reason
            for reason, condition in (
                (
                    "iterations_used < 6",
                    pd.isna(row["iterations_used"]) or row["iterations_used"] < 6,
                ),
                (
                    "stop_reason != max_iterations",
                    row["stop_reason"] != "max_iterations",
                ),
            )
            if condition
        ),
        axis=1,
    )
    exclusions = failing[
        [
            "dataset",
            "seed",
            "method",
            "iterations_used",
            "stop_reason",
            "exclusion_reason",
        ]
    ].sort_values(["dataset", "seed", "method"], ignore_index=True)

    if selected.empty:
        return (
            _paired_comparison_table(
                combined.loc[
                    combined["method"].isin(_available_baseline_methods(combined))
                ]
            ),
            exclusions,
            0,
        )
    paired_auc = selected.pivot_table(
        index=["dataset", "seed"],
        columns="method",
        values="test_auc",
        aggfunc="mean",
    )
    eligible_runs = selected.set_index(
        ["dataset", "seed", "method"]
    )["eligible"].unstack("method")
    if not set(AGENT_METHODS).issubset(paired_auc.columns) or not set(
        AGENT_METHODS
    ).issubset(eligible_runs.columns):
        complete = pd.DataFrame(
            columns=list(AGENT_METHODS),
            index=pd.MultiIndex.from_arrays(
                [[], []],
                names=["dataset", "seed"],
            ),
            dtype=float,
        )
        excluded_pair_count = 0
    else:
        paired_auc = paired_auc.dropna(subset=list(AGENT_METHODS))
        eligible_runs = eligible_runs.reindex(paired_auc.index)
        keep = eligible_runs[list(AGENT_METHODS)].all(axis=1)
        excluded_pair_count = int((~keep).sum())
        complete = paired_auc.loc[keep, list(AGENT_METHODS)]
    if complete.empty and not isinstance(complete.index, pd.MultiIndex):
        complete.index = pd.MultiIndex.from_arrays(
            [[], []],
            names=["dataset", "seed"],
        )
    eligible_keys = set(complete.index.tolist())
    agent_key_tuples = list(
        agents.loc[agents["method"].isin(AGENT_METHODS), ["dataset", "seed"]]
        .itertuples(index=False, name=None)
    )
    keep_agent_rows = [
        (dataset, seed) in eligible_keys for dataset, seed in agent_key_tuples
    ]
    eligible_agent_rows = agents.loc[agents["method"].isin(AGENT_METHODS)].loc[
        keep_agent_rows
    ]
    filtered_combined = pd.concat(
        [
            combined.loc[
                combined["method"].isin(_available_baseline_methods(combined))
            ],
            eligible_agent_rows[
                [
                    "dataset",
                    "method",
                    "seed",
                    "test_auc",
                    "best_cv_auc",
                    "compute_seconds",
                    "prompt_tokens",
                    "completion_tokens",
                    "llm_calls",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    comparisons = _paired_comparison_table(filtered_combined)
    comparisons.insert(0, "analysis", "eligible complete runs only")
    comparisons["excluded_complete_pairs"] = excluded_pair_count
    return comparisons, exclusions, excluded_pair_count


def _memory_plain_dataset_table(combined: pd.DataFrame) -> pd.DataFrame:
    paired = combined.loc[
        combined["method"].isin(("plain", "memory")),
        ["dataset", "seed", "method", "test_auc"],
    ].pivot_table(
        index=["dataset", "seed"],
        columns="method",
        values="test_auc",
        aggfunc="mean",
    )
    paired = paired.reindex(columns=["plain", "memory"]).dropna()
    paired["difference_memory_minus_plain"] = paired["memory"] - paired["plain"]
    means = paired.groupby(level="dataset")[["plain", "memory"]].mean()
    differences = paired["difference_memory_minus_plain"].unstack("seed")
    differences = differences.rename(
        columns={seed: f"difference_seed_{seed}" for seed in differences.columns}
    )
    table = means.join(differences, how="outer")
    table["difference_mean"] = table["memory"] - table["plain"]
    table = table.reset_index()
    return table.rename(
        columns={
            "plain": "plain_mean",
            "memory": "memory_mean",
        }
    )


def _paired_failure_tables(
    agents: pd.DataFrame,
    traces: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_frame = pd.DataFrame(traces)
    if trace_frame.empty:
        empty = pd.DataFrame()
        return empty, empty
    trace_frame["iteration"] = pd.to_numeric(trace_frame["iteration"], errors="coerce")
    attempts = trace_frame.loc[trace_frame["iteration"].gt(0)].copy()
    attempts["error_count"] = attempts["status"].eq("error").astype(int)
    attempts["noop_count"] = attempts["noop"].eq(True).astype(int)
    per_run = attempts.groupby(
        ["dataset", "seed", "method"], as_index=False
    )[["error_count", "noop_count"]].sum()
    pair_frames: list[pd.DataFrame] = []
    stats: list[dict[str, Any]] = []
    for measure in ("error_count", "noop_count"):
        paired = per_run.pivot_table(
            index=["dataset", "seed"],
            columns="method",
            values=measure,
            aggfunc="sum",
        ).reindex(columns=["plain", "memory"])
        paired = paired.dropna(subset=["plain", "memory"])
        paired[f"{measure}_difference_memory_minus_plain"] = (
            paired["memory"] - paired["plain"]
        )
        pair_frames.append(
            paired[["plain", "memory", f"{measure}_difference_memory_minus_plain"]]
            .rename(
                columns={
                    "plain": f"{measure}_plain",
                    "memory": f"{measure}_memory",
                    f"{measure}_difference_memory_minus_plain": f"{measure}_difference",
                }
            )
        )
        differences = paired[f"{measure}_difference_memory_minus_plain"]
        dataset_means = differences.groupby(level="dataset").mean()
        dataset_stat, dataset_p = _wilcoxon_summary(dataset_means)
        pair_stat, pair_p = _wilcoxon_summary(differences)
        stats.append(
            {
                "measure": measure,
                "paired_dataset_seed_n": len(differences),
                "dataset_n": len(dataset_means),
                "mean_difference_paired_dataset_seed": differences.mean(),
                "mean_difference_dataset_means": dataset_means.mean(),
                "wilcoxon_dataset_statistic": dataset_stat,
                "wilcoxon_dataset_p": dataset_p,
                "wilcoxon_pairs_statistic_non_independent": pair_stat,
                "wilcoxon_pairs_p_non_independent": pair_p,
            }
        )
    paired_detail = pair_frames[0].join(pair_frames[1], how="outer")
    per_dataset_rows: list[dict[str, Any]] = []
    for dataset, group in paired_detail.groupby(level="dataset"):
        row: dict[str, Any] = {
            "dataset": dataset,
            "paired_seeds": len(group),
        }
        for measure in ("error_count", "noop_count"):
            for column in (
                f"{measure}_plain",
                f"{measure}_memory",
                f"{measure}_difference",
            ):
                row[f"{column}_mean"] = group[column].mean()
        per_dataset_rows.append(row)
    per_dataset = pd.DataFrame(per_dataset_rows)
    return per_dataset, pd.DataFrame(stats)


def _run_cv_curves(
    traces: list[dict[str, Any]],
    agents: pd.DataFrame,
) -> tuple[dict[tuple[str, int, str], list[float]], set[tuple[str, int]]]:
    trace_frame = pd.DataFrame(traces)
    if trace_frame.empty:
        return {}, set()
    run_keys = set(
        agents[["dataset", "seed", "method"]]
        .itertuples(index=False, name=None)
    )
    run_curves: dict[tuple[str, int, str], list[float]] = {}
    for (dataset, seed, method), run in trace_frame.groupby(
        ["dataset", "seed", "method"], sort=True
    ):
        key = (dataset, int(seed), method)
        if key not in run_keys:
            continue
        run = run.copy()
        run["iteration"] = pd.to_numeric(run["iteration"], errors="coerce")
        run["cv_auc"] = pd.to_numeric(run["cv_auc"], errors="coerce")
        current_best = np.nan
        values: list[float] = []
        for iteration in range(7):
            candidates = run.loc[
                run["iteration"].eq(iteration)
                & run["status"].eq("ok")
                & run["noop"].eq(False),
                "cv_auc",
            ].dropna()
            if len(candidates):
                candidate_best = float(candidates.max())
                current_best = (
                    candidate_best
                    if not np.isfinite(current_best)
                    else max(float(current_best), candidate_best)
                )
            values.append(float(current_best) if np.isfinite(current_best) else np.nan)
        run_curves[key] = values

    pair_keys = {
        (dataset, seed)
        for dataset, seed, method in run_curves
        if method == "plain"
        and (dataset, seed, "memory") in run_curves
    }
    return run_curves, pair_keys


def _early_error_or_noop_pairs(
    traces: list[dict[str, Any]],
    pair_keys: set[tuple[str, int]],
) -> set[tuple[str, int]]:
    flagged: set[tuple[str, int]] = set()
    for record in traces:
        if (
            record["method"] in AGENT_METHODS
            and (record["dataset"], record["seed"]) in pair_keys
            and 1 <= int(record.get("iteration", 0)) <= 3
            and (
                record.get("status") == "error"
                or record.get("noop") is True
            )
        ):
            flagged.add((record["dataset"], int(record["seed"])))
    return pair_keys - flagged


def _mean_curve(
    run_curves: dict[tuple[str, int, str], list[float]],
    method: str,
    pair_keys: set[tuple[str, int]] | None = None,
) -> list[float]:
    keys = [
        key
        for key in run_curves
        if key[2] == method
        and (pair_keys is None or (key[0], key[1]) in pair_keys)
    ]
    means: list[float] = []
    for iteration in range(7):
        values = [run_curves[key][iteration] for key in keys]
        finite = [value for value in values if np.isfinite(value)]
        means.append(float(np.mean(finite)) if finite else np.nan)
    return means


def _early_progress(
    run_curves: dict[tuple[str, int, str], list[float]],
    pair_keys: set[tuple[str, int]],
    clean_pair_keys: set[tuple[str, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows = []
    for iteration in range(7):
        curve_rows.append(
            {
                "iteration": iteration,
                "plain_all_successful_non_noop": _mean_curve(
                    run_curves, "plain"
                )[iteration],
                "memory_all_successful_non_noop": _mean_curve(
                    run_curves, "memory"
                )[iteration],
                "plain_clean_pairs": _mean_curve(
                    run_curves, "plain", clean_pair_keys
                )[iteration],
                "memory_clean_pairs": _mean_curve(
                    run_curves, "memory", clean_pair_keys
                )[iteration],
                "all_paired_dataset_seeds": len(pair_keys),
                "clean_dataset_seeds": len(clean_pair_keys),
            }
        )
    curves = pd.DataFrame(curve_rows)

    difference_rows: list[dict[str, Any]] = []
    for iteration in (1, 2, 3, 6):
        differences = pd.Series(
            {
                (dataset, seed): run_curves[(dataset, seed, "memory")][iteration]
                - run_curves[(dataset, seed, "plain")][iteration]
                for dataset, seed in pair_keys
                if np.isfinite(run_curves[(dataset, seed, "memory")][iteration])
                and np.isfinite(run_curves[(dataset, seed, "plain")][iteration])
            },
            dtype=float,
        )
        dataset_differences = differences.groupby(level=0).mean()
        if dataset_differences.empty:
            ci_low = ci_high = np.nan
        else:
            rng = np.random.default_rng(BOOTSTRAP_SEED + iteration)
            samples = rng.choice(
                dataset_differences.to_numpy(dtype=float),
                size=(BOOTSTRAP_RESAMPLES, len(dataset_differences)),
                replace=True,
            ).mean(axis=1)
            ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
        difference_rows.append(
            {
                "iteration": iteration,
                "paired_dataset_seed_n": len(differences),
                "dataset_n": len(dataset_differences),
                "mean_difference_memory_minus_plain": differences.mean(),
                "dataset_mean_bootstrap_95_ci": (
                    f"[{ci_low:.6f}, {ci_high:.6f}]"
                    if np.isfinite(ci_low) and np.isfinite(ci_high)
                    else "n/a"
                ),
            }
        )
    return curves, pd.DataFrame(difference_rows)


def _selection_optimism(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available_methods = [
        method for method in METHODS if method in set(combined.get("method", []))
    ]
    for method in available_methods:
        method_runs = combined.loc[combined["method"].eq(method)]
        differences = method_runs["test_auc"] - method_runs["best_cv_auc"]
        differences = differences.dropna()
        rows.append(
            {
                "method": method,
                "runs_with_both_auc_values": len(differences),
                "mean_test_auc_minus_best_cv_auc": (
                    differences.mean() if len(differences) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _attempt_rates(
    agents: pd.DataFrame, traces: list[dict[str, Any]]
) -> pd.DataFrame:
    trace_frame = pd.DataFrame(traces)
    rows: list[dict[str, Any]] = []
    for method in AGENT_METHODS:
        method_runs = agents.loc[agents["method"].eq(method)]
        method_trace = (
            trace_frame.loc[trace_frame["method"].eq(method)].copy()
            if not trace_frame.empty
            else pd.DataFrame()
        )
        if not method_trace.empty:
            completed_run_keys = set(
                method_runs[["dataset", "seed"]]
                .dropna()
                .itertuples(index=False, name=None)
            )
            method_trace = method_trace.loc[
                [
                    (dataset, seed) in completed_run_keys
                    for dataset, seed in zip(
                        method_trace["dataset"],
                        method_trace["seed"],
                    )
                ]
            ]
        if not method_trace.empty:
            # Iteration zero is the deterministic baseline, not an LLM request.
            traced_attempts = method_trace.loc[
                pd.to_numeric(method_trace["iteration"], errors="coerce").gt(0)
            ]
        else:
            traced_attempts = pd.DataFrame()
        attempt_count = int(
            pd.to_numeric(
                method_runs.get("llm_calls", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
        categories: dict[str, pd.Series] = {
            "error": traced_attempts.get(
                "status", pd.Series(dtype=object)
            ).eq("error"),
            "invalid": traced_attempts.get(
                "status", pd.Series(dtype=object)
            ).eq("invalid_code"),
            "timeout": traced_attempts.get(
                "status", pd.Series(dtype=object)
            ).eq("timeout"),
            "noop": (
                traced_attempts.get(
                    "noop", pd.Series(False, index=traced_attempts.index)
                ).eq(True)
            ),
        }
        for category, flags in categories.items():
            attempts_in_category = int(flags.sum())
            result_column = {
                "error": "n_error",
                "invalid": "n_invalid",
                "timeout": "n_timeout",
                "noop": "n_noop_candidates",
            }[category]
            run_event_counts = pd.to_numeric(
                method_runs.get(result_column, pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0)
            runs_with_event = (
                int(run_event_counts.gt(0).sum()) if len(method_runs) else 0
            )
            rows.append(
                {
                    "method": method,
                    "event": category,
                    "event_attempts": attempts_in_category,
                    "all_llm_attempts": attempt_count,
                    "per_attempt_rate": (
                        attempts_in_category / attempt_count
                        if attempt_count
                        else np.nan
                    ),
                    "runs_with_at_least_one": runs_with_event,
                    "agent_runs": len(method_runs),
                    "runs_with_at_least_one_rate": (
                        runs_with_event / len(method_runs)
                        if len(method_runs)
                        else np.nan
                    ),
                    "mean_failed_iterations_per_run": (
                        pd.to_numeric(
                            method_runs.get(
                                "n_failed_iterations",
                                pd.Series(dtype=float),
                            ),
                            errors="coerce",
                        ).fillna(0).mean()
                        if len(method_runs)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _attempt_quality(traces: list[dict[str, Any]]) -> pd.DataFrame:
    baseline_auc: dict[tuple[str, int, str], float] = {}
    attempts_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in AGENT_METHODS
    }
    for record in traces:
        method = record.get("method")
        if method not in AGENT_METHODS:
            continue
        try:
            iteration = int(record.get("iteration", 0))
            seed = int(record["seed"])
            cv_auc = float(record["cv_auc"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(record["dataset"]), seed, method)
        if iteration == 0:
            baseline_auc[key] = cv_auc
        elif iteration >= 1:
            attempts_by_method[method].append(record)

    rows: list[dict[str, Any]] = []
    for method in AGENT_METHODS:
        attempts = attempts_by_method[method]
        qualified: list[float] = []
        for record in attempts:
            try:
                key = (str(record["dataset"]), int(record["seed"]), method)
                candidate_auc = float(record["cv_auc"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                record.get("status") == "ok"
                and record.get("noop") is False
                and key in baseline_auc
            ):
                qualified.append(candidate_auc - baseline_auc[key])
        improved = sum(difference > 0.002 for difference in qualified)
        rows.append(
            {
                "method": method,
                "all_llm_attempts": len(attempts),
                "ok_non_noop_attempts_with_baseline": len(qualified),
                "mean_cv_auc_delta_vs_iteration_0": (
                    float(np.mean(qualified)) if qualified else np.nan
                ),
                "share_qualified_attempts_delta_gt_0_002": (
                    improved / len(qualified) if qualified else np.nan
                ),
                "share_all_attempts_ok_non_noop_delta_gt_0_002": (
                    improved / len(attempts) if attempts else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _classifier_distribution(
    traces: list[dict[str, Any]],
) -> pd.DataFrame:
    first_attempts: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in traces:
        method = record.get("method")
        if method not in (*AGENT_METHODS, "zoo"):
            continue
        try:
            iteration = int(record.get("iteration", -1))
            seed = int(record["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        if iteration != 1 or not record.get("code"):
            continue
        key = (str(record["dataset"]), method, seed)
        if key not in first_attempts:
            first_attempts[key] = record
    classifier_suffixes = (
        "Classifier",
        "Regression",
        "Regressor",
        "SVC",
        "NB",
        "DiscriminantAnalysis",
        "Perceptron",
    )
    counts: dict[tuple[str, str], int] = {}
    for (_, method, _), record in first_attempts.items():
        classes = [
            tag
            for tag in _change_tags(str(record["code"]))
            if tag.endswith(classifier_suffixes)
        ]
        classifier = ", ".join(classes) if classes else "(not identified)"
        counts[(method, classifier)] = counts.get((method, classifier), 0) + 1
    method_counts: dict[str, int] = {}
    for (method, _), count in counts.items():
        method_counts[method] = method_counts.get(method, 0) + count
    rows = [
        {
            "method": method,
            "classifier_class": classifier,
            "runs": count,
            "share_of_iteration_1_first_attempts": (
                count / method_counts[method] if method_counts[method] else np.nan
            ),
        }
        for (method, classifier), count in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def _cost_table(
    combined: pd.DataFrame, traces: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_frame = pd.DataFrame(traces)
    if not trace_frame.empty:
        trace_frame["llm_seconds"] = pd.to_numeric(
            trace_frame.get("llm_seconds"), errors="coerce"
        )
        trace_frame["iteration"] = pd.to_numeric(
            trace_frame.get("iteration"), errors="coerce"
        )
    rows: list[dict[str, Any]] = []
    slow_attempts: list[dict[str, Any]] = []
    available_methods = [
        method for method in METHODS if method in set(combined.get("method", []))
    ]
    for method in available_methods:
        runs = combined.loc[combined["method"].eq(method)]
        cost = {
            "method": method,
            "runs": len(runs),
            "compute_seconds_mean": (
                runs["compute_seconds"].mean() if len(runs) else np.nan
            ),
            "compute_seconds_median": (
                runs["compute_seconds"].median() if len(runs) else np.nan
            ),
            "total_prompt_tokens": (
                runs["prompt_tokens"].sum() if len(runs) else np.nan
            ),
            "total_completion_tokens": (
                runs["completion_tokens"].sum() if len(runs) else np.nan
            ),
            "mean_llm_calls": runs["llm_calls"].mean() if len(runs) else np.nan,
            "median_per_attempt_llm_seconds": np.nan,
            "attempts_over_600_seconds": 0,
            "over_600_seconds_flag": "NO",
        }
        if method in AGENT_METHODS and not trace_frame.empty:
            method_attempts = trace_frame.loc[
                trace_frame["method"].eq(method)
                & trace_frame["iteration"].gt(0)
                & trace_frame["llm_seconds"].gt(0)
            ]
            seconds = method_attempts["llm_seconds"].dropna()
            cost["median_per_attempt_llm_seconds"] = (
                seconds.median() if len(seconds) else np.nan
            )
            long_attempts = method_attempts.loc[
                method_attempts["llm_seconds"].gt(MAX_ATTEMPT_SECONDS)
            ]
            cost["attempts_over_600_seconds"] = len(long_attempts)
            cost["over_600_seconds_flag"] = "YES" if len(long_attempts) else "NO"
            for _, attempt in long_attempts.iterrows():
                slow_attempts.append(
                    {
                        "method": method,
                        "dataset": attempt["dataset"],
                        "seed": attempt["seed"],
                        "iteration": attempt["iteration"],
                        "attempt": attempt.get("attempt"),
                        "llm_seconds": attempt["llm_seconds"],
                    }
                )
        rows.append(cost)
    cost_table = pd.DataFrame(rows)
    autogluon_median = cost_table.loc[
        cost_table["method"].eq("autogluon"), "compute_seconds_median"
    ]
    denominator = (
        float(autogluon_median.iloc[0])
        if len(autogluon_median) and pd.notna(autogluon_median.iloc[0])
        else np.nan
    )
    cost_table["median_compute_ratio_vs_autogluon"] = np.where(
        denominator > 0,
        cost_table["compute_seconds_median"] / denominator,
        np.nan,
    )
    return cost_table, pd.DataFrame(
        slow_attempts,
        columns=(
            "method",
            "dataset",
            "seed",
            "iteration",
            "attempt",
            "llm_seconds",
        ),
    )


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data available._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _write_curves(curves: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves_path = OUTPUT_DIR / "curves.csv"
    curves.to_csv(curves_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    if not curves.empty:
        for column, label in (
            ("plain_all_successful_non_noop", "plain"),
            ("memory_all_successful_non_noop", "memory"),
        ):
            if column in curves.columns:
                axis.plot(
                    curves["iteration"],
                    curves[column],
                    marker="o",
                    label=label,
                )
    axis.set(
        title="Mean best-so-far training CV AUC by iteration",
        xlabel="Iteration",
        ylabel="Mean best-so-far CV AUC",
        xticks=range(7),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "curves.png", dpi=160)
    plt.close(figure)


def _write_clean_curves(curves: pd.DataFrame) -> None:
    curves.to_csv(OUTPUT_DIR / "curves_clean.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    plot_columns = (
        ("plain_all_successful_non_noop", "plain (all runs)", "-"),
        ("memory_all_successful_non_noop", "memory (all runs)", "-"),
        ("plain_clean_pairs", "plain (clean pairs)", "--"),
        ("memory_clean_pairs", "memory (clean pairs)", "--"),
    )
    for column, label, linestyle in plot_columns:
        if column in curves.columns:
            axis.plot(
                curves["iteration"],
                curves[column],
                marker="o",
                linestyle=linestyle,
                label=label,
            )
    axis.set(
        title="Successful, non-noop best-so-far CV AUC",
        xlabel="Iteration",
        ylabel="Mean best-so-far CV AUC",
        xticks=range(7),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "curves_clean.png", dpi=160)
    plt.close(figure)


def _write_test_auc_chart(per_dataset: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    if not per_dataset.empty:
        per_dataset.plot(kind="bar", ax=axis, width=0.82)
    axis.set(
        title="Mean test AUC by dataset and method",
        xlabel="Dataset",
        ylabel="Mean test ROC-AUC",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Method")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_auc_by_dataset.png", dpi=160)
    plt.close(figure)


def _write_memory_plain_dataset_table(table: pd.DataFrame) -> None:
    table.to_csv(OUTPUT_DIR / "memory_plain_by_dataset.csv", index=False)


def _build_report(
    per_dataset: pd.DataFrame,
    mean_ranks: pd.Series,
    comparisons: pd.DataFrame,
    attempt_rates: pd.DataFrame,
    costs: pd.DataFrame,
    slow_attempts: pd.DataFrame,
    curves: pd.DataFrame,
    trace_count: int,
    pairing_note: str,
    sensitivity: pd.DataFrame,
    excluded_runs: pd.DataFrame,
    excluded_pair_count: int,
    memory_plain_by_dataset: pd.DataFrame,
    failure_dataset_means: pd.DataFrame,
    failure_tests: pd.DataFrame,
    clean_pair_count: int,
    progress_differences: pd.DataFrame,
    selection_optimism: pd.DataFrame,
    attempt_quality: pd.DataFrame,
    classifier_distribution: pd.DataFrame,
) -> str:
    ranks = mean_ranks.rename("mean_rank").rename_axis("method").reset_index()
    if not slow_attempts.empty:
        slow_text = _markdown_table(slow_attempts)
    else:
        slow_text = "No individual LLM attempt exceeded 600 seconds."
    if excluded_runs.empty:
        excluded_text = "No plain or memory runs met the exclusion criteria."
    else:
        excluded_text = _markdown_table(excluded_runs)
    inputs_description = (
        "Inputs: `results/agent_results.csv`, `results/baselines.csv`, "
        f"and live traces under `results/traces/` ({trace_count} trace records). "
    )
    if ZOO_PATH.exists():
        inputs_description += "`results/zoo_results.csv` is included when present. "
    if ZOO_TRACE_DIR.exists():
        inputs_description += "Live zoo candidate traces are read from `results/zoo_traces/`. "
    inputs_description += "All generated files are in `results/analysis/`."
    return "\n\n".join(
        (
            "# Tabular agent experiment analysis",
            inputs_description,
            "## Mean test AUC by dataset and method",
            _markdown_table(per_dataset.reset_index()),
            "## Mean rank across datasets (1 = best)",
            _markdown_table(ranks),
            "## Paired test-AUC comparisons",
            (
                "Difference is agent minus comparator. Dataset-level confidence "
                "intervals use 10,000 bootstrap resamples of datasets with fixed "
                f"seed {BOOTSTRAP_SEED}. The paired-run Wilcoxon result is explicitly "
                "labeled non-independent because seeds from one dataset are not "
                "independent datasets. "
                f"{pairing_note}"
            ),
            _markdown_table(comparisons),
            "### Plain vs memory test AUC by dataset",
            (
                "Differences are memory minus plain; per-seed columns are individual "
                "paired differences."
            ),
            _markdown_table(memory_plain_by_dataset),
            "## Sensitivity: exclude incomplete or early-stopped agent pairs",
            (
                "Each listed run excludes its entire dataset-seed pair from all "
                "agent comparisons. Comparisons are recomputed using only complete "
                "dataset-seed "
                "pairs where both runs have `iterations_used >= 6` and "
                "`stop_reason == max_iterations`; baseline observations remain "
                "available for the eligible agent dataset-seeds. "
                f"Excluded complete pairs: {excluded_pair_count}."
            ),
            "### Runs meeting an exclusion criterion",
            excluded_text,
            "### Sensitivity comparison table",
            _markdown_table(sensitivity),
            "## Paired failures: errors and noops",
            (
                "Per-dataset values are means over paired seeds. Differences are "
                "memory minus plain. The dataset-level Wilcoxon uses dataset means; "
                "the seed-pair Wilcoxon is labeled non-independent."
            ),
            _markdown_table(failure_dataset_means),
            _markdown_table(failure_tests),
            "## Agent failure rates",
            (
                "`per_attempt_rate` divides category attempts by all LLM candidate "
                "attempts (the deterministic iteration-0 baseline is excluded). "
                "`runs_with_at_least_one_rate` is the fraction of agent runs with "
                "one or more attempts in that category. Mean failed iterations is "
                "reported per run."
            ),
            _markdown_table(attempt_rates),
            "## Cost by method",
            (
                "Baselines have zero prompt/completion tokens and LLM calls by "
                "definition; per-attempt LLM seconds are available only for agents. "
                "Mean and median compute seconds are adjacent columns."
            ),
            _markdown_table(costs),
            "## Selection optimism",
            (
                "Mean test AUC minus the selected run's best CV AUC. AutoGluon "
                "does not record a comparable best CV AUC in its baseline results."
            ),
            _markdown_table(selection_optimism),
            "### Individual attempts over 600 seconds",
            slow_text,
            "## Per-attempt candidate quality",
            (
                "For each live plain/memory attempt at iterations 1+, quality is "
                "candidate CV AUC minus that run's iteration-0 CV AUC. The final "
                "share uses all attempts as its denominator; the preceding share "
                "uses only successful, non-noop attempts with a recorded baseline."
            ),
            _markdown_table(attempt_quality),
            "## Classifier class at iteration 1",
            (
                "One first attempt per run is counted. Class names are extracted "
                "with the same tag parser used by experiment memory."
            ),
            _markdown_table(classifier_distribution),
            "## CV learning curves",
            (
                "Only successful, non-noop candidates can improve the running best; "
                "failed/noop attempts carry the previous score forward. The clean "
                f"subset contains {clean_pair_count} dataset-seed pairs with neither "
                "method having an error or noop at iterations 1-3. "
                + (
                    "No pairs qualify, so clean-series values are unavailable."
                    if clean_pair_count == 0
                    else ""
                )
            ),
            _markdown_table(curves),
            "### Paired memory-minus-plain CV differences",
            "Dataset-level bootstrap confidence intervals use 10,000 dataset resamples.",
            _markdown_table(progress_differences),
        )
    ) + "\n"


def main() -> int:
    baselines, agents, combined = _load_results()
    traces = _load_live_traces()
    zoo_traces = _load_live_zoo_traces()

    per_dataset = _per_dataset_auc(combined)
    mean_ranks = _mean_ranks(per_dataset)
    comparisons = _paired_comparison_table(combined)
    sensitivity, excluded_runs, excluded_pair_count = _full_iteration_sensitivity(
        agents, combined
    )
    attempt_rates = _attempt_rates(agents, traces)
    costs, slow_attempts = _cost_table(combined, traces)
    attempt_quality = _attempt_quality(traces)
    classifier_distribution = _classifier_distribution(traces + zoo_traces)
    memory_plain_by_dataset = _memory_plain_dataset_table(combined)
    failure_dataset_means, failure_tests = _paired_failure_tables(agents, traces)
    run_curves, paired_curve_keys = _run_cv_curves(traces, agents)
    clean_pair_keys = _early_error_or_noop_pairs(traces, paired_curve_keys)
    curves, progress_differences = _early_progress(
        run_curves,
        paired_curve_keys,
        clean_pair_keys,
    )
    selection_optimism = _selection_optimism(combined)
    plain_keys = set(
        agents.loc[agents["method"].eq("plain"), ["dataset", "seed"]]
        .itertuples(index=False, name=None)
    )
    memory_keys = set(
        agents.loc[agents["method"].eq("memory"), ["dataset", "seed"]]
        .itertuples(index=False, name=None)
    )
    missing_memory_pairs = sorted(plain_keys - memory_keys)
    pairing_note = (
        f"Complete plain/memory pairs available: {len(plain_keys & memory_keys)} "
        f"of {len(plain_keys)} plain runs."
    )
    if missing_memory_pairs:
        pairing_note += " Memory result rows missing for: " + ", ".join(
            f"{dataset} seed {seed}" for dataset, seed in missing_memory_pairs
        ) + ". These pairs are excluded; no AUC is imputed."

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_curves(curves)
    _write_clean_curves(curves)
    _write_test_auc_chart(per_dataset)
    _write_memory_plain_dataset_table(memory_plain_by_dataset)
    report = _build_report(
        per_dataset,
        mean_ranks,
        comparisons,
        attempt_rates,
        costs,
        slow_attempts,
        curves,
        len(traces),
        pairing_note,
        sensitivity,
        excluded_runs,
        excluded_pair_count,
        memory_plain_by_dataset,
        failure_dataset_means,
        failure_tests,
        len(clean_pair_keys),
        progress_differences,
        selection_optimism,
        attempt_quality,
        classifier_distribution,
    )
    report_path = OUTPUT_DIR / "summary.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {report_path}")
    print(f"Wrote {OUTPUT_DIR / 'curves.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'curves.png'}")
    print(f"Wrote {OUTPUT_DIR / 'curves_clean.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'curves_clean.png'}")
    print(f"Wrote {OUTPUT_DIR / 'test_auc_by_dataset.png'}")
    print(f"Wrote {OUTPUT_DIR / 'memory_plain_by_dataset.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
