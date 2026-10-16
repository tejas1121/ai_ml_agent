"""End-to-end Day 2 smoke test for the LLM client and candidate sandbox."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.code_template import CODE_TEMPLATE, extract_code, validate_code
from src.llm import LLMClient, ProviderResponse
from src.profiler import profile_dataset, profile_to_text
from src.sandbox import run_candidate, _training_data

MOCK_CODE = '''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model(X_train, y_train):
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", RandomForestClassifier(n_estimators=100, n_jobs=6, random_state=0)),
        ]
    )
'''

INFINITE_LOOP_CODE = '''def build_model(X_train, y_train):
    while True:
        pass
'''

MEMORY_HOG_CODE = '''def build_model(X_train, y_train):
    chunks = []
    while True:
        chunks.append(bytearray(50 * 1024 * 1024))
'''

RUNTIME_ERROR_CODE = '''def build_model(X_train, y_train):
    raise RuntimeError("intentional smoke-test failure")
'''


class _MockProvider:
    def chat(self, messages, temperature, max_tokens, seed) -> ProviderResponse:
        del messages, temperature, max_tokens, seed
        return ProviderResponse(MOCK_CODE, 0, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip Ollama and use a hard-coded valid estimator function.",
    )
    args = parser.parse_args()

    X_train, y_train = _training_data("credit-g", 0)
    profile_text = profile_to_text(profile_dataset(X_train, y_train))
    messages = [
        {"role": "system", "content": CODE_TEMPLATE},
        {
            "role": "user",
            "content": (
                "Build a binary classification model for this training profile:\n"
                f"{profile_text}"
            ),
        },
    ]

    if args.mock:
        with tempfile.TemporaryDirectory(prefix="tabagent-mock-llm-") as temp_dir:
            temp_path = Path(temp_dir)
            client = LLMClient(
                _provider=_MockProvider(),
                _cache_dir=temp_path / "llm_cache",
                _log_db_path=temp_path / "llm_log.db",
            )
            first_started = time.perf_counter()
            first = client.chat(messages, run_id="day2-smoke-credit-g-seed-0")
            first_call_seconds = time.perf_counter() - first_started
            second_started = time.perf_counter()
            second = client.chat(messages, run_id="day2-smoke-credit-g-seed-0")
            second_call_seconds = time.perf_counter() - second_started
    else:
        client = LLMClient()
        try:
            first_started = time.perf_counter()
            first = client.chat(messages, run_id="day2-smoke-credit-g-seed-0")
            first_call_seconds = time.perf_counter() - first_started
            second_started = time.perf_counter()
            second = client.chat(messages, run_id="day2-smoke-credit-g-seed-0")
            second_call_seconds = time.perf_counter() - second_started
        except RuntimeError as error:
            print(f"LLM request failed: {error}", file=sys.stderr)
            return 1

    print(
        f"LLM first call: prompt_tokens={first.prompt_tokens}, "
        f"completion_tokens={first.completion_tokens}, "
        f"seconds={first.seconds:.3f}, cached={first.cached}"
    )
    print(
        f"LLM second call: prompt_tokens={second.prompt_tokens}, "
        f"completion_tokens={second.completion_tokens}, "
        f"seconds={second.seconds:.3f}, cached={second.cached}, "
        f"wall_seconds={second_call_seconds:.4f}"
    )
    if not args.mock:
        print(f"LLM first-call wall seconds: {first_call_seconds:.3f}")

    code = extract_code(first.text)
    valid, reason = validate_code(code)
    if not valid:
        print(f"Candidate validation failed: {reason}", file=sys.stderr)
        return 1
    candidate = run_candidate(code, "credit-g", 0)
    print(
        f"Candidate: status={candidate.status}, cv_auc={candidate.cv_auc}, "
        f"seconds={candidate.seconds:.2f}, peak_mem_mb={candidate.peak_mem_mb:.1f}"
    )
    if candidate.status != "ok":
        print(f"Candidate error: {candidate.error}", file=sys.stderr)

    for label, snippet, timeout_s in (
        ("infinite loop", INFINITE_LOOP_CODE, 5),
        ("memory hog", MEMORY_HOG_CODE, 120),
        ("runtime error", RUNTIME_ERROR_CODE, 30),
    ):
        result = run_candidate(
            snippet,
            "credit-g",
            0,
            timeout_s=timeout_s,
        )
        print(
            f"Bad snippet ({label}): status={result.status}, "
            f"seconds={result.seconds:.2f}, peak_mem_mb={result.peak_mem_mb:.1f}"
        )
        if result.error:
            print(result.error)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
