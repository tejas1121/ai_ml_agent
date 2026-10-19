"""Run the plain/build/memory experiment phases sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_agent import ScriptedMockProvider
from src.agent import run_agent
from src.llm import LLMClient
from src.memory import build_memory
from src import tracking

DEFAULT_DATASETS = (
    "credit-g",
    "credit-approval",
    "diabetes",
    "blood-transfusion-service-center",
    "ilpd",
    "wdbc",
    "kc1",
    "phoneme",
)


class MemoryPoolError(RuntimeError):
    """Raised when a memory run cannot start with the configured pool."""


def _print_batch_summary(counts: dict[str, int]) -> None:
    print(
        f"Batch summary: done={counts['done']}, failed={counts['failed']}, "
        f"skipped={counts['skipped']}"
    )


def _load_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _resolve_batch_paths(
    mock_llm: bool,
    results_file: str | Path | None,
    traces_dir: str | Path | None,
    memory_db: str | Path | None,
) -> tuple[Path, Path, Path]:
    mock_root = PROJECT_ROOT / "results" / "mock"
    result_value = results_file or (
        mock_root / "agent_results.csv"
        if mock_llm
        else "results/agent_results.csv"
    )
    traces_value = traces_dir or (
        mock_root / "traces" if mock_llm else "results/traces"
    )
    database_value = memory_db or (
        mock_root / "memory.db" if mock_llm else "data/memory.db"
    )
    results_path = _project_path(result_value).resolve()
    trace_path = _project_path(traces_value).resolve()
    database_path = _project_path(database_value).resolve()

    real_paths = {
        "traces-dir": (PROJECT_ROOT / "results" / "traces").resolve(),
        "memory-db": (PROJECT_ROOT / "data" / "memory.db").resolve(),
        "results-file": (PROJECT_ROOT / "results" / "agent_results.csv").resolve(),
    }
    selected_paths = {
        "traces-dir": trace_path,
        "memory-db": database_path,
        "results-file": results_path,
    }
    if mock_llm:
        for name, path in selected_paths.items():
            if path == real_paths[name]:
                raise ValueError(f"Mock mode refuses to use the real {name}: {path}")
            if not _is_within(path, mock_root):
                raise ValueError(
                    f"Mock mode requires {name} to be under {mock_root}: {path}"
                )
    else:
        forbidden_roots = (
            (PROJECT_ROOT / "results" / "mock").resolve(),
            (PROJECT_ROOT / "scripts" / "fixtures").resolve(),
        )
        for name in ("traces-dir", "memory-db"):
            path = selected_paths[name]
            if any(_is_within(path, root) for root in forbidden_roots):
                raise ValueError(
                    f"Live mode refuses {name} under results/mock or scripts/fixtures: "
                    f"{path}"
                )
    return results_path, trace_path, database_path


def _read_finished(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as result_file:
        reader = csv.DictReader(result_file)
        if not reader.fieldnames or not {"dataset", "method", "seed"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Results file {path} must contain dataset, method, and seed columns."
            )
        return {
            (row["dataset"], row["method"], int(row["seed"]))
            for row in reader
            if row.get("dataset") and row.get("method") and row.get("seed") is not None
        }


def _check_ollama(config: dict[str, Any]) -> None:
    llm_config = config["llm"]
    base_url = str(llm_config.get("base_url", "http://localhost:11434")).rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Ollama is not responding at {base_url}. Start Ollama and verify "
            "its local API before running a live batch."
        ) from error


def _check_memory_pool(database: Path) -> None:
    message = (
        f"Memory pool not found at {database.resolve()}. "
        "Run scripts/build_memory.py after the plain phase."
    )
    if not database.is_file():
        raise MemoryPoolError(message)
    try:
        import sqlite3
        from contextlib import closing

        with closing(
            sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
        ) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory'"
            ).fetchone()
    except sqlite3.Error as error:
        raise MemoryPoolError(message) from error
    if exists is None:
        raise MemoryPoolError(message)


def _count_live_plain_pairs(traces_dir: Path) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for trace_path in traces_dir.rglob("*.jsonl") if traces_dir.exists() else ():
        try:
            header = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
        except (IndexError, json.JSONDecodeError):
            continue
        if header.get("mode") != "live" or header.get("method") != "plain":
            continue
        suffix = "_plain_"
        if suffix not in trace_path.stem:
            continue
        dataset, seed_text = trace_path.stem.rsplit(suffix, 1)
        try:
            pairs.add((dataset, int(seed_text)))
        except ValueError:
            continue
    return pairs


def _append_batch_error(
    dataset: str,
    method: str,
    seed: int,
    error: BaseException,
    mock_llm: bool,
) -> None:
    log_path = PROJECT_ROOT / "results" / (
        Path("mock") / "batch_errors.log" if mock_llm else Path("batch_errors.log")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            f"dataset={dataset} method={method} seed={seed}\n"
        )
        log_file.write("".join(traceback.format_exception(error)))


def _print_finished(result: dict[str, Any]) -> None:
    print(
        f"Finished dataset={result['dataset']} method={result['method']} "
        f"seed={result['seed']} best_cv_auc={result['best_cv_auc']} "
        f"test_auc={result['test_auc']} "
        f"compute_seconds={result['compute_seconds']:.2f} "
        f"failures(invalid/timeout/memory/error/noop)="
        f"{result['n_invalid']}/{result['n_timeout']}/{result['n_memory']}/"
        f"{result['n_error']}/{result['n_noop_candidates']}"
    )


def _run_method(
    dataset: str,
    method: str,
    seed: int,
    results_path: Path,
    trace_dir: Path,
    memory_db: Path,
    mock_llm: bool,
) -> dict[str, Any]:
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    client: LLMClient | None = None
    mock_root = PROJECT_ROOT / "results" / "mock"
    if mock_llm:
        mock_root.mkdir(parents=True, exist_ok=True)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix=f"tabagent-batch-{dataset}-{method}-{seed}-",
            dir=mock_root,
        )
        temp_path = Path(temporary_directory.name)
        client = LLMClient(
            _provider=ScriptedMockProvider(),
            _cache_dir=temp_path / "cache",
            _log_db_path=temp_path / "llm_log.db",
        )
    try:
        # run_agent reads the same configured llm.seed for both methods.
        tracking_root = PROJECT_ROOT / "results" / "mock" if mock_llm else PROJECT_ROOT
        tracking_override = (
            patch.object(tracking, "PROJECT_ROOT", tracking_root)
            if mock_llm
            else nullcontext()
        )
        with tracking_override:
            return run_agent(
                dataset,
                seed,
                client=client,
                results_file=results_path,
                mode="mock" if mock_llm else "live",
                method=method,
                memory_db=memory_db,
                trace_dir=trace_dir,
            )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def _run_phase(
    phase: str,
    datasets: list[str],
    seeds: list[int],
    results_path: Path,
    trace_dir: Path,
    memory_db: Path,
    mock_llm: bool,
    allow_partial: bool,
    limit: int | None,
    counts: dict[str, int],
) -> None:
    if phase == "build":
        if mock_llm:
            # Mock traces are deliberately not valid memory sources. Use the offline,
            # live-format fixture traces to exercise this isolated test pool.
            source_traces = PROJECT_ROOT / "scripts" / "fixtures" / "traces"
            meta_file = PROJECT_ROOT / "scripts" / "fixtures" / "meta_features.json"
        else:
            source_traces = trace_dir
            meta_file = None
        live_pairs = _count_live_plain_pairs(source_traces)
        expected_pairs = 8 * 3
        if len(live_pairs) < expected_pairs and not allow_partial:
            raise RuntimeError(
                f"Memory build refused: found {len(live_pairs)} live plain "
                f"dataset-seed traces; {expected_pairs} required. Finish the plain "
                "phase or pass --allow-partial."
            )
        if not live_pairs:
            raise RuntimeError(f"No live plain traces found in {source_traces}.")
        entries = build_memory(
            source_traces,
            memory_db,
            meta_features_file=meta_file,
        )
        print(
            f"Build complete: {len(live_pairs)} live plain dataset-seed traces, "
            f"{entries} memory entries at {memory_db}"
        )
        return

    if phase == "memory":
        _check_memory_pool(memory_db)
    method = "plain" if phase == "plain" else "memory"
    # Seed-major ordering keeps every partial batch balanced across datasets.
    for seed in seeds:
        for dataset in datasets:
            if limit is not None and counts["done"] + counts["failed"] >= limit:
                print(f"Run limit reached ({limit}).")
                return
            existing = _read_finished(results_path)
            key = (dataset, method, seed)
            if key in existing:
                counts["skipped"] += 1
                continue
            print(f"Starting dataset={dataset} method={method} seed={seed}")
            try:
                result = _run_method(
                    dataset,
                    method,
                    seed,
                    results_path,
                    trace_dir,
                    memory_db,
                    mock_llm,
                )
            except Exception as error:
                _append_batch_error(dataset, method, seed, error, mock_llm)
                counts["failed"] += 1
                print(
                    f"Failed dataset={dataset} method={method} seed={seed}: "
                    f"{type(error).__name__}: {error}"
                )
                if isinstance(error, FileNotFoundError) and str(error).startswith(
                    "Memory pool not found"
                ):
                    raise MemoryPoolError(str(error)) from error
                continue
            counts["done"] += 1
            _print_finished(result)


def main() -> int:
    config = _load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("plain", "build", "memory", "all"),
        default="all",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=config.get("datasets", list(DEFAULT_DATASETS)),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--results-file")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--traces-dir")
    parser.add_argument("--memory-db")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative.")
    try:
        results_path, trace_dir, memory_db = _resolve_batch_paths(
            args.mock_llm,
            args.results_file,
            args.traces_dir,
            args.memory_db,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    counts = {"done": 0, "failed": 0, "skipped": 0}
    if args.phase == "memory":
        try:
            _check_memory_pool(memory_db)
        except MemoryPoolError as error:
            print(str(error), file=sys.stderr)
            _print_batch_summary(counts)
            return 1
    if not args.mock_llm:
        try:
            _check_ollama(config)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            _print_batch_summary(counts)
            return 1
    print(
        f"LLM base seed {config['llm'].get('seed', 0)} is shared by plain and "
        "memory runs for each matching dataset/seed."
    )

    phases = (
        ("plain", "build", "memory")
        if args.phase == "all"
        else (args.phase,)
    )
    try:
        for phase in phases:
            _run_phase(
                phase,
                args.datasets,
                args.seeds,
                results_path,
                trace_dir,
                memory_db,
                args.mock_llm,
                args.allow_partial,
                args.limit,
                counts,
            )
    except MemoryPoolError as error:
        print(str(error), file=sys.stderr)
        _print_batch_summary(counts)
        return 1
    except KeyboardInterrupt:
        print("\nBatch interrupted. Completed rows are saved; rerun to resume.")
        _print_batch_summary(counts)
        return 130
    except Exception as error:
        print(f"Batch stopped: {type(error).__name__}: {error}", file=sys.stderr)
        _print_batch_summary(counts)
        return 1

    _print_batch_summary(counts)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
