"""Build and retrieve training-only experiment memory from plain-agent traces."""

from __future__ import annotations

import ast
import json
import math
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

from src.profiler import meta_features, profile_dataset
from src.train_only import load_train_only

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_NAME = re.compile(r"(?P<dataset>.+)_plain_(?P<seed>\d+)\.jsonl$")
TAG_NAME = re.compile(
    r"[A-Za-z_]\w*(?:Classifier|Regression|Regressor|SVC|NB|"
    r"DiscriminantAnalysis|Perceptron|Scaler|Encoder)\b"
)


def _database_path(db_path: str | Path) -> Path:
    path = Path(db_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _reject_mock_directory(traces_dir: Path) -> None:
    parts = tuple(part.lower() for part in traces_dir.resolve().parts)
    if any(
        parts[index : index + 2] == ("results", "mock")
        for index in range(len(parts) - 1)
    ):
        raise ValueError("Mock trace directories cannot be used to build experiment memory.")


def _trace_files(traces_dir: Path) -> list[Path]:
    _reject_mock_directory(traces_dir)
    if not traces_dir.exists():
        return []
    if not traces_dir.is_dir():
        raise NotADirectoryError(f"Trace path is not a directory: {traces_dir}")

    files = sorted(traces_dir.rglob("*.jsonl"))
    for path in files:
        parts = tuple(part.lower() for part in path.resolve().parts)
        if any(
            parts[index : index + 2] == ("results", "mock")
            for index in range(len(parts) - 1)
        ):
            raise ValueError(f"Mock trace file cannot be used for memory: {path}")
    return files


def _read_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or not isinstance(records[0], dict):
        return {}, []

    header = records[0]
    if header.get("type") == "header" and header.get("mode") == "mock":
        raise ValueError(f"Mock trace header cannot be used for memory: {path}")
    if header.get("type") != "header":
        return {}, []
    if not all(isinstance(record, dict) for record in records[1:]):
        raise ValueError(f"Trace rows must be JSON objects: {path}")
    return header, records[1:]


def count_live_plain_traces(traces_dir: str | Path) -> int:
    """Count explicitly identified live plain-agent trace files."""
    directory = Path(traces_dir)
    return sum(
        1
        for path in _trace_files(directory)
        if (
            (header := _read_trace(path)[0]).get("mode") == "live"
            and header.get("method") == "plain"
        )
    )


def _dataset_meta_features(dataset: str, seed: int) -> list[float]:
    X_train, y_train = load_train_only(dataset, seed)
    return meta_features(profile_dataset(X_train, y_train))


def _change_tags(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return sorted(set(TAG_NAME.findall(code)))

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.rsplit(".", 1)[-1]
                for alias in node.names
            )
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return sorted(name for name in names if TAG_NAME.fullmatch(name))


def build_memory(
    traces_dir: str | Path,
    db_path: str | Path = "data/memory.db",
    meta_features_file: str | Path | None = None,
) -> int:
    """Rebuild memory from live plain traces, optionally using supplied profiles."""
    directory = Path(traces_dir)
    trace_files = _trace_files(directory)
    database = _database_path(db_path).resolve()
    real_memory_database = (PROJECT_ROOT / "data" / "memory.db").resolve()
    fixture_directory = (PROJECT_ROOT / "scripts" / "fixtures").resolve()
    if (
        database == real_memory_database
        and trace_files
        and (
            _is_within(directory, fixture_directory)
            or any(_is_within(trace_path, fixture_directory) for trace_path in trace_files)
        )
    ):
        raise ValueError(
            f"Refusing to build {real_memory_database} from scripts/fixtures traces."
        )

    supplied_features: dict[str, list[float]] | None = None
    if meta_features_file is not None:
        features_path = Path(meta_features_file)
        if not features_path.is_absolute():
            features_path = PROJECT_ROOT / features_path
        with features_path.open(encoding="utf-8") as features_file:
            raw_features = json.load(features_file)
        if not isinstance(raw_features, dict):
            raise ValueError("Meta-features file must contain a JSON object.")
        expected_length = len(
            meta_features(
                {
                    "n_rows": 0.0,
                    "n_cols": 0.0,
                    "n_numeric": 0.0,
                    "n_categorical": 0.0,
                    "missing_rate": 0.0,
                    "class_balance": 0.0,
                    "mean_abs_skewness": 0.0,
                    "mean_categorical_cardinality": 0.0,
                }
            )
        )
        supplied_features = {}
        for dataset_name, vector in raw_features.items():
            if not isinstance(dataset_name, str) or not isinstance(vector, list):
                raise ValueError(
                    "Meta-features must map dataset names to numeric vectors."
                )
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Meta-features for {dataset_name!r} must be numeric."
                ) from error
            if len(values) != expected_length or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(
                    f"Meta-features for {dataset_name!r} must contain "
                    f"{expected_length} finite values."
                )
            supplied_features[dataset_name] = values

    features_by_run: dict[tuple[str, int], list[float]] = {}
    rows: list[tuple[Any, ...]] = []
    for trace_path in trace_files:
        header, records = _read_trace(trace_path)
        if header.get("mode") == "mock":
            raise ValueError(f"Mock trace header cannot be used for memory: {trace_path}")
        if header.get("mode") != "live" or header.get("method") != "plain":
            continue
        match = TRACE_NAME.fullmatch(trace_path.name)
        if match is None:
            raise ValueError(
                f"Live plain trace filename must be '<dataset>_plain_<seed>.jsonl': "
                f"{trace_path.name}"
            )
        dataset = match.group("dataset")
        seed = int(match.group("seed"))
        baseline_scores = [
            float(record["cv_auc"])
            for record in records
            if record.get("iteration") == 0
            and record.get("status") == "ok"
            and record.get("cv_auc") is not None
        ]
        baseline_auc = baseline_scores[0] if baseline_scores else None
        run_key = (dataset, seed)
        if run_key not in features_by_run:
            if supplied_features is None:
                features_by_run[run_key] = _dataset_meta_features(dataset, seed)
            else:
                try:
                    features_by_run[run_key] = supplied_features[dataset].copy()
                except KeyError as error:
                    raise ValueError(
                        f"No meta-features supplied for dataset {dataset!r}."
                    ) from error

        for record in records:
            iteration = record.get("iteration")
            status = record.get("status")
            if iteration == 0 or status == "infra_error":
                continue
            cv_auc = record.get("cv_auc")
            cv_auc = float(cv_auc) if cv_auc is not None else None
            delta = (
                cv_auc - baseline_auc
                if cv_auc is not None and baseline_auc is not None
                else None
            )
            noop = bool(record.get("noop", False))
            success = (
                status == "ok"
                and not noop
                and delta is not None
                and delta > 0.002
            )
            rows.append(
                (
                    dataset,
                    seed,
                    int(iteration),
                    int(record["attempt"]),
                    str(record.get("hypothesis", "")),
                    json.dumps(_change_tags(str(record.get("code", "")))),
                    str(status),
                    cv_auc,
                    delta,
                    int(success),
                    int(noop),
                    json.dumps(features_by_run[run_key], separators=(",", ":")),
                )
            )

    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("DROP TABLE IF EXISTS memory")
            connection.execute(
                """
                CREATE TABLE memory (
                    dataset TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    iteration INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    hypothesis TEXT NOT NULL,
                    change_tags TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cv_auc REAL,
                    delta_vs_baseline REAL,
                    success INTEGER NOT NULL,
                    noop INTEGER NOT NULL,
                    dataset_meta_features TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO memory (
                    dataset, seed, iteration, attempt, hypothesis, change_tags,
                    status, cv_auc, delta_vs_baseline, success, noop,
                    dataset_meta_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    return len(rows)


def retrieve(
    target_dataset: str,
    target_meta: Sequence[float],
    k_datasets: int = 3,
    n_best: int = 2,
    n_worst: int = 1,
    db_path: str | Path = "data/memory.db",
) -> list[dict[str, Any]]:
    """Retrieve nearest-dataset experiments while enforcing leave-one-out."""
    if k_datasets < 0 or n_best < 0 or n_worst < 0:
        raise ValueError("Retrieval limits must be non-negative.")
    target_vector = [float(value) for value in target_meta]
    if not all(math.isfinite(value) for value in target_vector):
        raise ValueError("Target meta-features must be finite numbers.")

    database = _database_path(db_path).resolve()
    missing_pool_error = (
        f"Memory pool not found at {database}. "
        "Run scripts/build_memory.py after the plain phase."
    )
    if not database.is_file():
        raise FileNotFoundError(missing_pool_error)

    database_uri = f"{database.as_uri()}?mode=ro"
    try:
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            memory_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory'"
            ).fetchone()
            if memory_table is None:
                raise FileNotFoundError(missing_pool_error)
            pool = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT dataset, seed, iteration, attempt, hypothesis, change_tags,
                           status, cv_auc, delta_vs_baseline, success, noop,
                           dataset_meta_features
                    FROM memory
                    """
                )
            ]
    except sqlite3.OperationalError as error:
        if not database.is_file() or "no such table: memory" in str(error).lower():
            raise FileNotFoundError(missing_pool_error) from error
        raise

    # The dataset exclusion must stay here so every caller gets leave-one-out.
    pool = [entry for entry in pool if entry["dataset"] != target_dataset]
    if not pool or k_datasets == 0:
        return []

    vectors_by_dataset: dict[str, list[list[float]]] = defaultdict(list)
    for entry in pool:
        vector = json.loads(entry["dataset_meta_features"])
        if len(vector) != len(target_vector):
            raise ValueError(
                f"Meta-feature length mismatch for dataset {entry['dataset']!r}."
            )
        vectors_by_dataset[entry["dataset"]].append(
            [float(value) for value in vector]
        )
        entry["change_tags"] = json.loads(entry["change_tags"])
        entry["success"] = bool(entry["success"])
        entry["noop"] = bool(entry["noop"])

    dataset_vectors = {
        dataset: [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(target_vector))
        ]
        for dataset, vectors in vectors_by_dataset.items()
    }
    vectors = list(dataset_vectors.values())
    means = [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(len(target_vector))
    ]
    standard_deviations = [
        math.sqrt(
            sum((vector[index] - means[index]) ** 2 for vector in vectors)
            / len(vectors)
        )
        for index in range(len(target_vector))
    ]
    standard_deviations = [
        deviation if deviation else 1.0 for deviation in standard_deviations
    ]

    distances = {
        dataset: math.sqrt(
            sum(
                (
                    (vector[index] - target_vector[index])
                    / standard_deviations[index]
                )
                ** 2
                for index in range(len(target_vector))
            )
        )
        for dataset, vector in dataset_vectors.items()
    }
    nearest_datasets = sorted(distances, key=lambda item: (distances[item], item))[
        :k_datasets
    ]

    selected: list[dict[str, Any]] = []
    for dataset in nearest_datasets:
        dataset_entries = [entry for entry in pool if entry["dataset"] == dataset]
        successful = sorted(
            (entry for entry in dataset_entries if entry["success"]),
            key=lambda entry: (
                -(entry["delta_vs_baseline"] or 0.0),
                entry["seed"],
                entry["iteration"],
                entry["attempt"],
            ),
        )[:n_best]
        unsuccessful = sorted(
            (
                entry
                for entry in dataset_entries
                if not entry["success"]
                and entry["delta_vs_baseline"] is not None
                and entry["delta_vs_baseline"] < 0
            ),
            key=lambda entry: (
                entry["delta_vs_baseline"],
                entry["seed"],
                entry["iteration"],
                entry["attempt"],
            ),
        )[:n_worst]
        if not unsuccessful:
            unsuccessful = sorted(
                (
                    entry
                    for entry in dataset_entries
                    if not entry["success"] and entry["cv_auc"] is None
                ),
                key=lambda entry: (
                    entry["seed"],
                    entry["iteration"],
                    entry["attempt"],
                ),
            )[:n_worst]
        selected.extend(successful)
        selected.extend(unsuccessful)
    return selected


def format_memory_context(
    entries: Sequence[dict[str, Any]],
    max_chars: int = 1800,
) -> str:
    """Format measured results without code or free-text opinions."""
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative.")
    if not entries or max_chars == 0:
        return ""

    lines = [
        "Measured results on OTHER datasets only; results may not transfer."
    ]
    for entry in entries:
        meta = json.loads(entry["dataset_meta_features"])
        tags = ", ".join(entry["change_tags"])
        delta = entry["delta_vs_baseline"]
        status = entry["status"]
        if delta is None and status in {"error", "invalid_code"}:
            result = "failed to run (code error)"
        elif delta is None and status in {"timeout", "memory"}:
            result = f"failed to run ({status})"
        elif delta is None:
            result = "no CV score"
        elif delta < 0:
            result = (
                f"scored lower than baseline (CV AUC {float(delta):+0.3f})"
            )
        elif delta > 0:
            result = (
                f"scored higher than baseline (CV AUC {float(delta):+0.3f})"
            )
        else:
            result = "scored equal to baseline (CV AUC +0.000)"
        lines.append(
            f"Similar dataset (rows={round(meta[0])}, "
            f"categorical={round(meta[3])}): tried [{tags}] -> {result}."
        )
    return "\n".join(lines)[:max_chars]
