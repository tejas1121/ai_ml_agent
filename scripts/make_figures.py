"""Create supplementary figures from frozen result CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_020
METHOD_ORDER = (
    "xgboost_default",
    "autogluon",
    "plain",
    "memory",
    "zoo",
)


def _read_results() -> pd.DataFrame:
    baselines_path = RESULTS_DIR / "baselines.csv"
    agents_path = RESULTS_DIR / "agent_results.csv"
    zoo_path = RESULTS_DIR / "zoo_results.csv"
    for path in (baselines_path, agents_path, zoo_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen result file not found: {path}")

    baselines = pd.read_csv(baselines_path)
    baselines = baselines.loc[
        baselines["method"].isin(("xgboost_default", "autogluon"))
    ].copy()
    if "status" in baselines.columns:
        baselines = baselines.loc[baselines["status"].eq("completed")]
    baselines["compute_seconds"] = pd.to_numeric(
        baselines["seconds"], errors="coerce"
    )
    baseline_results = baselines[
        ["dataset", "method", "test_auc", "compute_seconds"]
    ]

    agents = pd.read_csv(agents_path)
    if "mode" in agents.columns:
        agents = agents.loc[agents["mode"].eq("live")]
    agents = agents.loc[agents["method"].isin(("plain", "memory"))]
    agent_results = agents[
        ["dataset", "method", "test_auc", "compute_seconds"]
    ]

    zoo = pd.read_csv(zoo_path)
    if "mode" in zoo.columns:
        zoo = zoo.loc[zoo["mode"].eq("live")]
    zoo = zoo.assign(method="zoo")
    zoo_results = zoo[["dataset", "method", "test_auc", "compute_seconds"]]

    combined = pd.concat(
        (baseline_results, agent_results, zoo_results),
        ignore_index=True,
    )
    for column in ("test_auc", "compute_seconds"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.dropna(
        subset=["dataset", "method", "test_auc", "compute_seconds"]
    )
    return combined


def _summary_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    random = np.random.default_rng(BOOTSTRAP_SEED)
    for method in METHOD_ORDER:
        method_rows = results.loc[results["method"].eq(method)]
        dataset_means = (
            method_rows.groupby("dataset", sort=True)["test_auc"].mean()
        )
        if dataset_means.empty:
            raise ValueError(f"No valid frozen result rows for method {method!r}.")
        values = dataset_means.to_numpy(dtype=float)
        bootstrap_means = random.choice(
            values,
            size=(BOOTSTRAP_RESAMPLES, len(values)),
            replace=True,
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append(
            {
                "method": method,
                "runs": int(len(method_rows)),
                "datasets": int(len(values)),
                "median_compute_seconds": float(
                    method_rows["compute_seconds"].median()
                ),
                "mean_test_auc": float(values.mean()),
                "bootstrap_95_ci_low": float(ci_low),
                "bootstrap_95_ci_high": float(ci_high),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    return pd.DataFrame(rows)


def _write_figure(summary: pd.DataFrame) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 6))
    label_offsets = {
        "xgboost_default": (8, 8),
        "autogluon": (8, 8),
        "plain": (8, 8),
        "memory": (8, -14),
        "zoo": (8, 8),
    }
    for row in summary.itertuples(index=False):
        mean_auc = float(row.mean_test_auc)
        lower_error = mean_auc - float(row.bootstrap_95_ci_low)
        upper_error = float(row.bootstrap_95_ci_high) - mean_auc
        axis.errorbar(
            float(row.median_compute_seconds),
            mean_auc,
            yerr=np.array([[lower_error], [upper_error]]),
            fmt="o",
            capsize=4,
            markersize=7,
        )
        axis.annotate(
            row.method,
            (float(row.median_compute_seconds), mean_auc),
            xytext=label_offsets[row.method],
            textcoords="offset points",
        )

    axis.set_xscale("log")
    axis.set(
        title="Test AUC versus median compute cost",
        xlabel="Median compute_seconds per run (seconds, log scale)",
        ylabel="Mean test AUC across dataset means",
    )
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(ANALYSIS_DIR / "cost_vs_auc.png", dpi=160)
    plt.close(figure)


def main() -> int:
    results = _read_results()
    summary = _summary_table(results)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ANALYSIS_DIR / "cost_vs_auc.csv", index=False)
    _write_figure(summary)
    print(summary.to_string(index=False))
    print(f"Wrote {ANALYSIS_DIR / 'cost_vs_auc.csv'}")
    print(f"Wrote {ANALYSIS_DIR / 'cost_vs_auc.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
