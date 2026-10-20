"""Run the fixed no-LLM candidate zoo sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zoo import CandidateResult, run_zoo_dataset

RESULT_COLUMNS = (
    "dataset",
    "method",
    "seed",
    "best_cv_auc",
    "test_auc",
    "best_candidate",
    "n_candidates_ok",
    "compute_seconds",
    "mode",
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _validate_paths(results_file: Path, traces_dir: Path) -> None:
    protected_files = (
        PROJECT_ROOT / "results" / "agent_results.csv",
        PROJECT_ROOT / "results" / "baselines.csv",
        PROJECT_ROOT / "data" / "memory.db",
    )
    protected_trace_dir = PROJECT_ROOT / "results" / "traces"
    if any(
        results_file == path.resolve() or _within(results_file, path)
        for path in protected_files
    ):
        raise ValueError(f"Refusing to overwrite protected result path: {results_file}")
    if _within(results_file, protected_trace_dir):
        raise ValueError(f"Zoo results cannot be written under {protected_trace_dir}.")
    if _within(traces_dir, protected_trace_dir):
        raise ValueError(f"Zoo traces cannot be written under {protected_trace_dir}.")
    if any(
        traces_dir == path.resolve() or _within(traces_dir, path)
        for path in protected_files
    ):
        raise ValueError(f"Refusing to use protected output path: {traces_dir}")


def _completed_runs(results_file: Path) -> set[tuple[str, int]]:
    if not results_file.exists():
        return set()
    with results_file.open(newline="", encoding="utf-8") as csv_file:
        return {
            (row["dataset"], int(row["seed"]))
            for row in csv.DictReader(csv_file)
            if row.get("method") == "zoo" and row.get("dataset") and row.get("seed")
        }


def _append_result(results_file: Path, result: dict[str, Any]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_file.exists() or results_file.stat().st_size == 0
    with results_file.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: result[column] for column in RESULT_COLUMNS})


def _write_trace_header(path: Path, dataset: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "header",
        "mode": "live",
        "method": "zoo",
        "dataset": dataset,
        "seed": seed,
    }
    path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_candidate(path: Path, candidate: CandidateResult) -> None:
    record = {
        "type": "candidate",
        "mode": "live",
        "method": "zoo",
        "iteration": candidate.candidate_index,
        "candidate_index": candidate.candidate_index,
        "candidate_name": candidate.candidate_name,
        "attempt": 1,
        "status": candidate.status,
        "cv_auc": candidate.cv_auc,
        "best_so_far": None,
        "code": candidate.code,
        "error": candidate.error,
        "noop": False,
        "llm_seconds": 0.0,
        "sandbox_seconds": candidate.seconds,
    }
    with path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> int:
    config = _load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=config["datasets"])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=config["seeds"]
    )
    parser.add_argument(
        "--results-file",
        default="results/zoo_results.csv",
        help="CSV output path (defaults to results/zoo_results.csv).",
    )
    parser.add_argument(
        "--traces-dir",
        default="results/zoo_traces",
        help="Per-candidate trace directory (never results/traces).",
    )
    args = parser.parse_args()

    results_file = _project_path(args.results_file)
    traces_dir = _project_path(args.traces_dir)
    try:
        _validate_paths(results_file, traces_dir)
    except ValueError as error:
        parser.error(str(error))

    completed = _completed_runs(results_file)
    done = skipped = failed = 0
    for seed in args.seeds:
        for dataset in args.datasets:
            key = (dataset, seed)
            if key in completed:
                skipped += 1
                print(f"SKIP dataset={dataset} method=zoo seed={seed}")
                continue

            trace_path = traces_dir / f"{dataset}_zoo_{seed}.jsonl"
            _write_trace_header(trace_path, dataset, seed)
            try:
                result = run_zoo_dataset(
                    dataset,
                    seed,
                    on_candidate=lambda candidate: _append_candidate(
                        trace_path, candidate
                    ),
                )
                _append_result(results_file, result)
                completed.add(key)
                done += 1
                print(
                    f"dataset={dataset} method=zoo seed={seed} "
                    f"best_cv_auc={result['best_cv_auc']:.6f} "
                    f"test_auc={result['test_auc']:.6f} "
                    f"best_candidate={result['best_candidate']} "
                    f"n_candidates_ok={result['n_candidates_ok']} "
                    f"compute_seconds={result['compute_seconds']:.2f}"
                )
            except Exception:
                failed += 1
                error_text = traceback.format_exc()
                with trace_path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(
                        json.dumps(
                            {
                                "type": "run_error",
                                "mode": "live",
                                "method": "zoo",
                                "dataset": dataset,
                                "seed": seed,
                                "error": error_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(f"FAILED dataset={dataset} method=zoo seed={seed}")
                print(error_text)

    print(f"Zoo batch summary: done={done}, failed={failed}, skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
