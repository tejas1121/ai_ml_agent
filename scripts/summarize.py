"""Write analysis tables and CV curves from saved baseline and agent results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "results" / "baselines.csv"
AGENT_PATH = PROJECT_ROOT / "results" / "agent_results.csv"
TRACE_DIR = PROJECT_ROOT / "results" / "traces"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
METHODS = ("xgboost_default", "autogluon", "plain", "memory")
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
    if not baselines.empty:
        baselines = baselines.copy()
        _numeric(
            baselines,
            ("seed", "test_auc", "cv_auc", "seconds"),
        )
        baselines["method"] = baselines["method"].astype(str)
        baselines["compute_seconds"] = baselines.get("seconds", np.nan)
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

    baseline_columns = [
        "dataset",
        "method",
        "seed",
        "test_auc",
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
        "compute_seconds",
        "prompt_tokens",
        "completion_tokens",
        "llm_calls",
    ]
    frames = [
        frame[columns]
        for frame, columns in (
            (baselines, baseline_columns),
            (agents, agent_columns),
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


def _per_dataset_auc(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=list(METHODS))
    return combined.pivot_table(
        index="dataset",
        columns="method",
        values="test_auc",
        aggfunc="mean",
    ).reindex(columns=METHODS)


def _mean_ranks(per_dataset: pd.DataFrame) -> pd.Series:
    if per_dataset.empty:
        return pd.Series(index=METHODS, dtype=float)
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
    for agent_method in AGENT_METHODS:
        for baseline_method in BASELINE_METHODS:
            pair_data = combined.loc[
                combined["method"].isin((agent_method, baseline_method)),
                ["dataset", "seed", "method", "test_auc"],
            ]
            if pair_data.empty:
                paired = pd.DataFrame(columns=[agent_method, baseline_method])
            else:
                paired = pair_data.pivot_table(
                    index=["dataset", "seed"],
                    columns="method",
                    values="test_auc",
                    aggfunc="mean",
                )
                paired = paired.reindex(columns=[agent_method, baseline_method])
                paired = paired.dropna(subset=[agent_method, baseline_method])

            rows.append(_paired_comparison_row(paired, agent_method, baseline_method))
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
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Compare plain/memory only on complete pairs where both ran all iterations."""
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
            pd.DataFrame(),
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
        return pd.DataFrame(), exclusions, 0
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
    comparison = pd.DataFrame(
        [_paired_comparison_row(complete, "memory", "plain")]
    )
    comparison.insert(0, "analysis", "both runs >=6 iterations and max_iterations")
    comparison["excluded_complete_pairs"] = excluded_pair_count
    return comparison, exclusions, excluded_pair_count


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


def _curve_data(traces: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trace_frame = pd.DataFrame(traces)
    if trace_frame.empty:
        return pd.DataFrame(columns=["iteration", *AGENT_METHODS])

    for method in AGENT_METHODS:
        method_traces = trace_frame.loc[trace_frame["method"].eq(method)]
        run_curves: list[list[float]] = []
        for _, run in method_traces.groupby(["dataset", "seed"], sort=True):
            run = run.copy()
            run["iteration"] = pd.to_numeric(run["iteration"], errors="coerce")
            run["best_so_far"] = pd.to_numeric(run["best_so_far"], errors="coerce")
            current_best = np.nan
            curve: list[float] = []
            for iteration in range(7):
                records = run.loc[run["iteration"].eq(iteration)]
                scores = records["best_so_far"].dropna()
                iteration_failed = (
                    records                    .get(
                        "iteration_failed",
                        pd.Series(False, index=records.index),
                    )
                    .eq(True)
                    .any()
                )
                if len(scores) and not iteration_failed:
                    score = float(scores.max())
                    current_best = (
                        score if not np.isfinite(current_best) else max(current_best, score)
                    )
                curve.append(float(current_best) if np.isfinite(current_best) else np.nan)
            run_curves.append(curve)
        for iteration in range(7):
            values = [curve[iteration] for curve in run_curves]
            finite_values = [value for value in values if np.isfinite(value)]
            rows.append(
                {
                    "iteration": iteration,
                    "method": method,
                    "mean_best_cv_auc": (
                        float(np.mean(finite_values)) if finite_values else np.nan
                    ),
                    "runs": len(finite_values),
                }
            )
    long_frame = pd.DataFrame(rows)
    if long_frame.empty:
        return pd.DataFrame(columns=["iteration", *AGENT_METHODS])
    return long_frame.pivot(
        index="iteration",
        columns="method",
        values="mean_best_cv_auc",
    ).reindex(index=range(7), columns=AGENT_METHODS).reset_index()


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
    for method in METHODS:
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
            "mean_prompt_tokens": (
                runs["prompt_tokens"].mean() if len(runs) else np.nan
            ),
            "mean_completion_tokens": (
                runs["completion_tokens"].mean() if len(runs) else np.nan
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
    return pd.DataFrame(rows), pd.DataFrame(
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
        for method in AGENT_METHODS:
            if method in curves.columns:
                axis.plot(
                    curves["iteration"],
                    curves[method],
                    marker="o",
                    label=method,
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
    return "\n\n".join(
        (
            "# Tabular agent experiment analysis",
            (
                "Inputs: `results/agent_results.csv`, `results/baselines.csv`, "
                f"and live traces under `results/traces/` ({trace_count} trace records). "
                "All generated files are in `results/analysis/`."
            ),
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
            "## Sensitivity: exclude incomplete or early-stopped agent pairs",
            (
                "This repeats memory minus plain using only complete dataset-seed "
                "pairs where both runs have `iterations_used >= 6` and "
                "`stop_reason == max_iterations`. "
                f"Excluded complete pairs: {excluded_pair_count}."
            ),
            _markdown_table(sensitivity),
            "### Runs meeting an exclusion criterion",
            excluded_text,
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
            "### Individual attempts over 600 seconds",
            slow_text,
            "## CV learning curves",
            (
                "Mean best-so-far training CV AUC for iterations 0-6. Failed or "
                "missing iterations carry the preceding best score forward. "
                "`curves.csv` and `curves.png` contain the plotted series."
            ),
            _markdown_table(curves),
        )
    ) + "\n"


def main() -> int:
    baselines, agents, combined = _load_results()
    traces = _load_live_traces()

    per_dataset = _per_dataset_auc(combined)
    mean_ranks = _mean_ranks(per_dataset)
    comparisons = _paired_comparison_table(combined)
    sensitivity, excluded_runs, excluded_pair_count = _full_iteration_sensitivity(
        agents
    )
    attempt_rates = _attempt_rates(agents, traces)
    costs, slow_attempts = _cost_table(combined, traces)
    curves = _curve_data(traces)
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
    )
    report_path = OUTPUT_DIR / "summary.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {report_path}")
    print(f"Wrote {OUTPUT_DIR / 'curves.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
