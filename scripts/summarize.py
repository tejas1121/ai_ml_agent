"""Summarize baseline and agent experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "results" / "baselines.csv"
AGENT_PATH = PROJECT_ROOT / "results" / "agent_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary.csv"
METHODS = ("xgboost_default", "autogluon", "plain", "memory")
FAILURE_COLUMNS = {
    "error": "n_error",
    "invalid": "n_invalid",
    "timeout": "n_timeout",
    "memory": "n_memory",
    "noop": "n_noop_candidates",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baselines = _read_csv(BASELINE_PATH)
    agents = _read_csv(AGENT_PATH)
    if not baselines.empty:
        baselines = baselines.copy()
        baselines["compute_seconds"] = pd.to_numeric(
            baselines.get("seconds", 0), errors="coerce"
        )
        baselines["tokens"] = 0
        baselines["method"] = baselines["method"].astype(str)
        baselines["test_auc"] = pd.to_numeric(
            baselines.get("test_auc"), errors="coerce"
        )
    if not agents.empty:
        agents = agents.copy()
        _as_numeric(
            agents,
            [
                "test_auc",
                "compute_seconds",
                "prompt_tokens",
                "completion_tokens",
                *FAILURE_COLUMNS.values(),
            ],
        )
        agents["tokens"] = agents.get("prompt_tokens", 0).fillna(0) + agents.get(
            "completion_tokens", 0
        ).fillna(0)
    frames = [frame for frame in (baselines, agents) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not combined.empty:
        combined["seed"] = pd.to_numeric(combined["seed"], errors="coerce")
    return baselines, agents, combined


def _mean_auc_and_ranks(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if combined.empty:
        return pd.DataFrame(columns=list(METHODS)), pd.Series(dtype=float)
    per_dataset = combined.pivot_table(
        index="dataset",
        columns="method",
        values="test_auc",
        aggfunc="mean",
    ).reindex(columns=METHODS)
    ranks = per_dataset.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0, skipna=True)
    return per_dataset, mean_ranks


def _paired_wilcoxon(agents: pd.DataFrame) -> str:
    if agents.empty or not {"dataset", "seed", "method", "test_auc"}.issubset(
        agents.columns
    ):
        message = "Warning: Wilcoxon skipped; fewer than 5 memory/plain pairs (0 found)."
        print(message)
        return message
    paired = agents.loc[
        agents["method"].isin(("plain", "memory")),
        ["dataset", "seed", "method", "test_auc"],
    ].pivot_table(
        index=["dataset", "seed"],
        columns="method",
        values="test_auc",
        aggfunc="mean",
    )
    if not {"plain", "memory"}.issubset(paired.columns):
        paired = pd.DataFrame(columns=["plain", "memory"])
    paired = paired.dropna(subset=["plain", "memory"])
    if len(paired) < 5:
        message = (
            f"Warning: Wilcoxon skipped; fewer than 5 memory/plain pairs "
            f"({len(paired)} found)."
        )
        print(message)
        return message
    try:
        statistic, p_value = wilcoxon(paired["memory"], paired["plain"])
    except ValueError as error:
        message = f"Warning: Wilcoxon could not be computed: {error}"
        print(message)
        return message
    message = (
        f"Wilcoxon memory vs plain: n={len(paired)}, "
        f"statistic={statistic:.4f}, p-value={p_value:.6g}"
    )
    print(message)
    return message


def _failure_rates(
    agents: pd.DataFrame,
    baselines: pd.DataFrame,
    method: str,
) -> dict[str, float]:
    if method in {"xgboost_default", "autogluon"}:
        method_runs = (
            baselines.loc[baselines["method"] == method]
            if not baselines.empty
            else baselines
        )
        failed_rate = (
            float((method_runs["status"] != "completed").mean())
            if len(method_runs) and "status" in method_runs.columns
            else 0.0
        )
        return {
            label: failed_rate if label == "error" else 0.0
            for label in FAILURE_COLUMNS
        }
    method_runs = agents.loc[agents["method"] == method] if not agents.empty else agents
    run_count = len(method_runs)
    rates: dict[str, float] = {}
    for label, column in FAILURE_COLUMNS.items():
        if run_count == 0 or column not in method_runs.columns:
            rates[label] = 0.0
        else:
            counts = pd.to_numeric(method_runs[column], errors="coerce").fillna(0)
            rates[label] = float((counts > 0).mean())
    return rates


def main() -> int:
    baselines, agents, combined = _load_results()
    per_dataset, mean_ranks = _mean_auc_and_ranks(combined)

    print("Mean test AUC per dataset and method:")
    print(per_dataset.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nMean rank across datasets (1 = best):")
    print(mean_ranks.to_string(float_format=lambda value: f"{value:.4f}"))
    wilcoxon_message = _paired_wilcoxon(agents)

    summary_rows: list[dict[str, Any]] = []
    for dataset, row in per_dataset.iterrows():
        for method in METHODS:
            value = row.get(method)
            summary_rows.append(
                {
                    "section": "dataset_auc",
                    "dataset": dataset,
                    "method": method,
                    "mean_test_auc": value,
                }
            )

    for method in METHODS:
        method_runs = combined.loc[combined["method"] == method] if not combined.empty else combined
        compute_seconds = (
            pd.to_numeric(method_runs.get("compute_seconds", 0), errors="coerce")
            .fillna(0)
            .sum()
            if len(method_runs)
            else 0.0
        )
        tokens = (
            pd.to_numeric(method_runs.get("tokens", 0), errors="coerce").fillna(0).sum()
            if len(method_runs)
            else 0
        )
        failures = _failure_rates(agents, baselines, method)
        print(
            f"{method}: runs={len(method_runs)}, compute_seconds={compute_seconds:.2f}, "
            f"tokens={int(tokens)}, failure_rates="
            + ", ".join(f"{name}={rate:.3f}" for name, rate in failures.items())
        )
        summary_rows.append(
            {
                "section": "method_summary",
                "method": method,
                "mean_rank": mean_ranks.get(method),
                "runs": len(method_runs),
                "compute_seconds": compute_seconds,
                "tokens": int(tokens),
                **{f"{name}_rate": rate for name, rate in failures.items()},
            }
        )

    summary_rows.append(
        {
            "section": "wilcoxon",
            "method": "memory_vs_plain",
            "wilcoxon": wilcoxon_message,
        }
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    print(f"\nWrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
