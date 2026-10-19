"""Run one plain agent experiment; no batch runner is provided."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import run_agent, run_is_finished
from src.llm import LLMClient, ProviderResponse

MOCK_VALID_CODE = '''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model(X_train, y_train):
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=80,
                    random_state=0,
                    n_jobs=6,
                ),
            ),
        ]
    )
'''


class ScriptedMockProvider:
    """Return one valid candidate, one broken response, then a valid repair."""

    def __init__(self) -> None:
        self.responses = [
            (
                "# HYPOTHESIS: A random forest can capture useful nonlinear feature interactions.\n"
                f"```python\n{MOCK_VALID_CODE}\n```"
            ),
            (
                "# HYPOTHESIS: A deliberately broken candidate verifies the repair path.\n"
                "```python\ndef build_model(:\n    return None\n```"
            ),
            (
                "# HYPOTHESIS: Repair the previous syntax failure with a valid forest pipeline.\n"
                f"```python\n{MOCK_VALID_CODE}\n```"
            ),
        ]
        self.call_index = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> ProviderResponse:
        del messages, temperature, max_tokens, seed
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
        else:
            response = (
                f"# HYPOTHESIS: Repeat a deterministic forest candidate for mock "
                f"call {self.call_index + 1}.\n"
                f"```python\n{MOCK_VALID_CODE}\n```"
            )
        self.call_index += 1
        return ProviderResponse(response, 120, 100)


def _resolve_results_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one TabAgent experiment.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", choices=("plain", "memory"), default="plain")
    parser.add_argument(
        "--memory-db",
        default="data/memory.db",
        help="SQLite experiment-memory database path, relative to the project root.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use a scripted valid/broken/repaired response sequence.",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="CSV result path, relative to the project root unless absolute.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.dataset):
        parser.error("dataset may contain only letters, digits, '.', '_' and '-'.")

    requested_results_file = args.results_file
    if requested_results_file is None:
        requested_results_file = (
            "results/mock/agent_results.csv"
            if args.mock_llm
            else "results/agent_results.csv"
        )
    results_path = _resolve_results_path(requested_results_file)
    default_live_results = (PROJECT_ROOT / "results" / "agent_results.csv").resolve()
    if args.mock_llm and results_path.resolve() == default_live_results:
        parser.error("Mock runs may not write to results/agent_results.csv.")

    mode = "mock" if args.mock_llm else "live"
    if run_is_finished(
        results_path,
        args.dataset,
        args.seed,
        args.method,
        mode,
    ):
        print(
            f"Skipping completed run: dataset={args.dataset}, "
            f"method={args.method}, seed={args.seed}"
        )
        return 0

    client = None
    mock_cache_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.mock_llm:
        mock_cache_directory = tempfile.TemporaryDirectory(prefix="tabagent-agent-mock-")
        client = LLMClient(
            _provider=ScriptedMockProvider(),
            _cache_dir=Path(mock_cache_directory.name) / "cache",
            _log_db_path=Path(mock_cache_directory.name) / "llm_log.db",
        )

    try:
        run_agent(
            args.dataset,
            args.seed,
            client=client,
            results_file=results_path,
            mode=mode,
            method=args.method,
            memory_db=_resolve_results_path(args.memory_db),
        )
    except RuntimeError as error:
        print(f"Agent run failed: {error}", file=sys.stderr)
        return 1
    finally:
        if mock_cache_directory is not None:
            mock_cache_directory.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
