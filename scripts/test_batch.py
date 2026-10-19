"""Offline tests for batch path isolation and memory-builder safeguards."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_batch
from src import memory


class BatchPathTests(unittest.TestCase):
    def test_mock_defaults_are_isolated_under_results_mock(self) -> None:
        results, traces, database = run_batch._resolve_batch_paths(
            True,
            None,
            None,
            None,
        )
        mock_root = (PROJECT_ROOT / "results" / "mock").resolve()
        for path in (results, traces, database):
            self.assertTrue(path.is_relative_to(mock_root))
        self.assertEqual(traces, mock_root / "traces")
        self.assertEqual(database, mock_root / "memory.db")

    def test_mock_refuses_real_paths_and_other_output_directories(self) -> None:
        real_paths = (
            ("results/agent_results.csv", None, None),
            (None, "results/traces", None),
            (None, None, "data/memory.db"),
            ("results/agent_results.csv", "results/traces", "data/memory.db"),
        )
        for results, traces, database in real_paths:
            with self.subTest(results=results, traces=traces, database=database):
                with self.assertRaisesRegex(ValueError, "Mock mode"):
                    run_batch._resolve_batch_paths(
                        True,
                        results,
                        traces,
                        database,
                    )

        with self.assertRaisesRegex(ValueError, "under"):
            run_batch._resolve_batch_paths(
                True,
                "results/mock/batch.csv",
                "results/mock/traces",
                "data/other.db",
            )

    def test_live_mode_refuses_mock_and_fixture_paths(self) -> None:
        for forbidden in ("results/mock", "scripts/fixtures"):
            for parameter in ("traces", "database"):
                arguments: dict[str, str | None] = {
                    "results": "results/agent_results.csv",
                    "traces": "results/traces",
                    "database": "data/memory.db",
                }
                arguments[parameter] = f"{forbidden}/unsafe"
                with self.subTest(forbidden=forbidden, parameter=parameter):
                    with self.assertRaisesRegex(ValueError, "Live mode refuses"):
                        run_batch._resolve_batch_paths(
                            False,
                            arguments["results"],
                            arguments["traces"],
                            arguments["database"],
                        )


class MemoryBuildSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tabagent-batch-test-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.project_root = self.root / "project"
        (self.project_root / "scripts" / "fixtures" / "traces").mkdir(
            parents=True
        )
        (self.project_root / "data").mkdir()
        self.fixture_trace = (
            self.project_root
            / "scripts"
            / "fixtures"
            / "traces"
            / "fixture-alpha_plain_0.jsonl"
        )

    def _write_trace(self, mode: str) -> None:
        records = [
            {"type": "header", "mode": mode, "method": "plain"},
            {
                "iteration": 0,
                "attempt": 0,
                "status": "ok",
                "cv_auc": 0.7,
            },
        ]
        self.fixture_trace.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_fixture_traces_cannot_write_to_real_memory_database(self) -> None:
        self._write_trace("live")
        database = self.project_root / "data" / "memory.db"
        with patch.object(memory, "PROJECT_ROOT", self.project_root):
            with self.assertRaisesRegex(ValueError, "scripts/fixtures"):
                memory.build_memory(
                    self.project_root / "scripts" / "fixtures" / "traces",
                    database,
                    meta_features_file=PROJECT_ROOT
                    / "scripts"
                    / "fixtures"
                    / "meta_features.json",
                )
        self.assertFalse(database.exists())

    def test_mock_trace_cannot_write_to_real_memory_database(self) -> None:
        mock_traces = self.root / "mock-input"
        mock_traces.mkdir()
        mock_trace = mock_traces / "fixture-alpha_plain_0.jsonl"
        records = [
            {"type": "header", "mode": "mock", "method": "plain"},
            {
                "iteration": 0,
                "attempt": 0,
                "status": "ok",
                "cv_auc": 0.7,
            },
        ]
        mock_trace.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        database = self.project_root / "data" / "memory.db"
        with patch.object(memory, "PROJECT_ROOT", self.project_root):
            with self.assertRaisesRegex(ValueError, "Mock trace header"):
                memory.build_memory(
                    mock_traces,
                    database,
                    meta_features_file=PROJECT_ROOT
                    / "scripts"
                    / "fixtures"
                    / "meta_features.json",
                )
        self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
