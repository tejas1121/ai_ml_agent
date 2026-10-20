"""Synthetic, offline tests for the training-only experiment-memory module."""

from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, contextmanager
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import memory
from src.agent import AgentState, propose
from src.llm import LLMResult

FIXTURE_TRACES = Path(__file__).resolve().parent / "fixtures" / "traces"
FIXTURE_META_FILE = Path(__file__).resolve().parent / "fixtures" / "meta_features.json"
META_FEATURES = json.loads(FIXTURE_META_FILE.read_text(encoding="utf-8"))


class _CapturingClient:
    def __init__(self) -> None:
        self.prompts: list[list[dict[str, str]]] = []

    def chat(self, messages, **kwargs) -> LLMResult:
        del kwargs
        self.prompts.append(messages)
        return LLMResult(
            "# HYPOTHESIS: A synthetic candidate.\n```python\npass\n```",
            1,
            1,
            0.0,
            False,
        )


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tabagent-memory-test-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "memory.db"
        self.row_count = memory.build_memory(
            FIXTURE_TRACES,
            self.db_path,
            meta_features_file=FIXTURE_META_FILE,
        )

    def test_fixture_meta_file_avoids_openml_dataset_loading(self) -> None:
        db_path = self.root / "meta-file-only.db"
        with patch.object(memory, "_dataset_meta_features") as meta_feature_loader:
            memory.build_memory(
                FIXTURE_TRACES,
                db_path,
                meta_features_file=FIXTURE_META_FILE,
            )
        meta_feature_loader.assert_not_called()

    def test_builder_uses_attempts_after_baseline_and_ignores_infra(self) -> None:
        self.assertEqual(self.row_count, 24)
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(
                "SELECT iteration, status, success, noop, change_tags "
                "FROM memory ORDER BY dataset, seed, iteration"
            ).fetchall()
        self.assertTrue(all(iteration > 0 for iteration, *_ in rows))
        self.assertFalse(any(status == "infra_error" for _, status, *_ in rows))
        noop_rows = [row for row in rows if row[3]]
        self.assertTrue(noop_rows)
        self.assertTrue(all(not row[2] for row in noop_rows))
        self.assertIn("ExtraTreesClassifier", json.loads(rows[0][4]))

    def test_leave_one_dataset_out_for_each_fixture_dataset(self) -> None:
        for dataset, target_meta in META_FEATURES.items():
            entries = memory.retrieve(
                dataset,
                target_meta,
                db_path=self.db_path,
            )
            self.assertTrue(entries)
            self.assertTrue(all(entry["dataset"] != dataset for entry in entries))
        limited = memory.retrieve(
            "fixture-alpha",
            META_FEATURES["fixture-alpha"],
            k_datasets=2,
            n_best=1,
            n_worst=1,
            db_path=self.db_path,
        )
        self.assertEqual(len({entry["dataset"] for entry in limited}), 2)
        for dataset in {entry["dataset"] for entry in limited}:
            dataset_entries = [
                entry for entry in limited if entry["dataset"] == dataset
            ]
            self.assertLessEqual(
                sum(entry["success"] for entry in dataset_entries),
                1,
            )
            self.assertLessEqual(
                sum(not entry["success"] for entry in dataset_entries),
                1,
            )
        example = memory.retrieve(
            "fixture-alpha",
            META_FEATURES["fixture-alpha"],
            db_path=self.db_path,
        )
        print("\nExample memory context:")
        print(memory.format_memory_context(example))

    def test_retrieval_prefers_scored_lower_entries_to_code_errors(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                UPDATE memory
                SET cv_auc = 0.673, delta_vs_baseline = -0.027
                WHERE dataset = 'fixture-alpha' AND seed = 0 AND iteration = 2
                """
            )
            connection.commit()
        entries = memory.retrieve(
            "fixture-beta",
            META_FEATURES["fixture-beta"],
            k_datasets=1,
            n_best=0,
            n_worst=1,
            db_path=self.db_path,
        )
        self.assertEqual(len(entries), 1)
        self.assertLess(entries[0]["delta_vs_baseline"], 0)
        context = memory.format_memory_context(entries)
        self.assertIn(
            "scored lower than baseline (CV AUC -0.027)",
            context,
        )
        self.assertNotIn("failed to run (code error)", context)

    def test_retrieval_uses_code_error_when_no_lower_score_exists(self) -> None:
        entries = memory.retrieve(
            "fixture-alpha",
            META_FEATURES["fixture-alpha"],
            k_datasets=1,
            n_best=0,
            n_worst=1,
            db_path=self.db_path,
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "error")
        self.assertIsNone(entries[0]["cv_auc"])
        self.assertIn(
            "failed to run (code error)",
            memory.format_memory_context(entries),
        )

    def test_missing_memory_file_raises_without_creating_it(self) -> None:
        missing_path = self.root / "not-built.db"
        expected = (
            f"Memory pool not found at {missing_path.resolve()}. "
            "Run scripts/build_memory.py after the plain phase."
        )
        with self.assertRaises(FileNotFoundError) as error:
            memory.retrieve(
                "fixture-alpha",
                META_FEATURES["fixture-alpha"],
                db_path=missing_path,
            )
        self.assertEqual(str(error.exception), expected)
        self.assertFalse(missing_path.exists())

    def test_database_without_memory_table_raises_clear_error(self) -> None:
        empty_database = self.root / "no-memory-table.db"
        with closing(sqlite3.connect(empty_database)) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
        expected = (
            f"Memory pool not found at {empty_database.resolve()}. "
            "Run scripts/build_memory.py after the plain phase."
        )
        with self.assertRaises(FileNotFoundError) as error:
            memory.retrieve(
                "fixture-alpha",
                META_FEATURES["fixture-alpha"],
                db_path=empty_database,
            )
        self.assertEqual(str(error.exception), expected)

    def test_existing_empty_memory_table_returns_empty_pool(self) -> None:
        empty_database = self.root / "empty-memory.db"
        with closing(sqlite3.connect(empty_database)) as connection:
            connection.execute(
                """
                CREATE TABLE memory (
                    dataset TEXT, seed INTEGER, iteration INTEGER, attempt INTEGER,
                    hypothesis TEXT, change_tags TEXT, status TEXT, cv_auc REAL,
                    delta_vs_baseline REAL, success INTEGER, noop INTEGER,
                    dataset_meta_features TEXT
                )
                """
            )
        entries = memory.retrieve(
            "fixture-alpha",
            META_FEATURES["fixture-alpha"],
            db_path=empty_database,
        )
        self.assertEqual(entries, [])
        self.assertEqual(memory.format_memory_context(entries), "")

    def test_memory_agent_does_not_continue_when_retrieval_fails(self) -> None:
        from src import agent

        @contextmanager
        def fake_agent_run(*args, **kwargs):
            del args, kwargs
            yield "test-run"

        def fake_profile(state: AgentState) -> None:
            state.memory_meta_features = META_FEATURES["fixture-alpha"]

        client = _CapturingClient()
        missing_database = self.root / "missing-agent-memory.db"
        with (
            patch.object(agent, "PROJECT_ROOT", self.root),
            patch.object(agent, "profile", side_effect=fake_profile),
            patch.object(agent, "agent_run", fake_agent_run),
        ):
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Memory pool not found",
            ):
                agent.run_agent(
                    "fixture-alpha",
                    0,
                    client=client,
                    method="memory",
                    memory_db=missing_database,
                )
        self.assertEqual(client.prompts, [])

    def test_empty_other_dataset_pool_keeps_plain_prompt_identical(self) -> None:
        only_one_dataset = self.root / "one-dataset-traces"
        only_one_dataset.mkdir()
        for seed in (0, 1):
            shutil.copy2(
                FIXTURE_TRACES / f"fixture-alpha_plain_{seed}.jsonl",
                only_one_dataset / f"fixture-alpha_plain_{seed}.jsonl",
            )
        one_dataset_db = self.root / "one-dataset.db"
        memory.build_memory(
            only_one_dataset,
            one_dataset_db,
            meta_features_file=FIXTURE_META_FILE,
        )
        entries = memory.retrieve(
            "fixture-alpha",
            META_FEATURES["fixture-alpha"],
            db_path=one_dataset_db,
        )
        self.assertEqual(entries, [])

        plain_client = _CapturingClient()
        memory_client = _CapturingClient()
        state = AgentState(
            dataset="fixture-alpha",
            seed=0,
            method="plain",
            profile_text="synthetic training profile",
        )
        propose(state, plain_client, "plain", base_seed=0)
        propose(
            state,
            memory_client,
            "memory",
            base_seed=0,
            memory_context=memory.format_memory_context(entries),
        )
        self.assertEqual(plain_client.prompts[0], memory_client.prompts[0])
        self.assertEqual(
            plain_client.prompts[0][1]["content"].encode("utf-8"),
            memory_client.prompts[0][1]["content"].encode("utf-8"),
        )

    def test_mock_directory_and_mock_header_are_refused(self) -> None:
        mock_dir = self.root / "results" / "mock" / "traces"
        mock_dir.mkdir(parents=True)
        shutil.copy2(
            FIXTURE_TRACES / "fixture-alpha_plain_0.jsonl",
            mock_dir / "fixture-alpha_plain_0.jsonl",
        )
        with self.assertRaises(ValueError):
            memory.build_memory(mock_dir, self.root / "mock-directory.db")

        mock_header_dir = self.root / "mock-header-traces"
        mock_header_dir.mkdir()
        rows = (FIXTURE_TRACES / "fixture-alpha_plain_0.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        header = json.loads(rows[0])
        header["mode"] = "mock"
        rows[0] = json.dumps(header)
        trace_path = mock_header_dir / "fixture-alpha_plain_0.jsonl"
        trace_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            memory.build_memory(mock_header_dir, self.root / "mock-header.db")

    def test_memory_schema_and_imports_have_no_held_out_score_source(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            columns = [
                row[1] for row in connection.execute("PRAGMA table_info(memory)")
            ]
        self.assertFalse(
            any("test" in column.lower() and "auc" in column.lower() for column in columns)
        )

        memory_source = (PROJECT_ROOT / "src" / "memory.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(memory_source)
        imported_modules = []
        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
                imported_names.extend(alias.name for alias in node.names)
        self.assertNotIn("src." + "evaluate_final", imported_modules)
        self.assertNotIn("load" + "_dataset", imported_names)

    def test_change_tags_are_classifier_scaler_and_encoder_names(self) -> None:
        tags = memory._change_tags(
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "model = LogisticRegression()\n"
        )
        self.assertEqual(
            tags,
            ["LogisticRegression", "OneHotEncoder", "StandardScaler"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
