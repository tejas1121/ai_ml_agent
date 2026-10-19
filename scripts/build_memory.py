"""Build the experiment-memory database from live plain-agent traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import build_memory, count_live_plain_traces


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build experiment memory from live plain-agent traces."
    )
    parser.add_argument(
        "--traces",
        "--traces-dir",
        dest="traces",
        default="results/traces",
        help="Plain-agent trace directory, relative to the project root.",
    )
    parser.add_argument(
        "--db",
        "--memory-db",
        dest="database",
        default="data/memory.db",
        help="SQLite database path, relative to the project root.",
    )
    parser.add_argument(
        "--meta-file",
        default=None,
        help="Optional JSON dataset-to-meta-feature mapping.",
    )
    args = parser.parse_args()
    traces_dir = Path(args.traces)
    database_path = Path(args.database)
    meta_features_path = Path(args.meta_file) if args.meta_file else None
    if not traces_dir.is_absolute():
        traces_dir = PROJECT_ROOT / traces_dir
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    if meta_features_path is not None and not meta_features_path.is_absolute():
        meta_features_path = PROJECT_ROOT / meta_features_path

    trace_count = count_live_plain_traces(traces_dir)
    if trace_count == 0:
        print(
            f"No live plain traces found in {traces_dir}; memory was not built.",
            file=sys.stderr,
        )
        return 1

    row_count = build_memory(
        traces_dir,
        database_path,
        meta_features_file=meta_features_path,
    )
    print(
        f"Built memory from {trace_count} live plain traces "
        f"({row_count} experiment entries): {database_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
