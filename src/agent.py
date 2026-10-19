"""Plain, no-memory LLM agent loop for tabular model candidates."""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
import yaml

from src.code_template import CODE_TEMPLATE, extract_code, validate_code
from src.evaluate_final import final_test_auc
from src.infra import is_infrastructure_error
from src.llm import LLMClient, LLMResult
from src.memory import format_memory_context, retrieve
from src.profiler import meta_features, profile_dataset, profile_to_text
from src.sandbox import SandboxResult, run_candidate
from src.train_only import load_train_only
from src.tracking import agent_run, log_agent_iteration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
HYPOTHESIS_PATTERN = re.compile(r"^# HYPOTHESIS:\s*(.+?)\s*$")
BUILD_MODEL_SKELETON = """Reference structure:
def build_model(X_train, y_train):
    numeric = X_train.select_dtypes(include=["number"]).columns
    categorical = X_train.select_dtypes(exclude=["number"]).columns
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", YourAllowedEstimator()),
    ])
Categorical columns are pandas category/object dtype; keep encoders in the
pipeline. Never use LabelEncoder or target encoders inside build_model.
LabelEncoder is for target columns only and breaks inside a Pipeline. For
categorical features use OneHotEncoder(handle_unknown='ignore') or
OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).
Refer to columns by name, never by integer position; target encoding or labeling
y is not needed. Use only the allowed imports listed above."""

DEFAULT_MODEL_CODE = '''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

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
                XGBClassifier(
                    random_state=SEED_VALUE,
                    n_jobs=6,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
'''


@dataclass
class AgentState:
    dataset: str
    seed: int
    method: str
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    best_code: str | None = None
    best_cv_auc: float | None = None
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "attempts": 0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "n_invalid": 0,
            "n_timeout": 0,
            "n_memory": 0,
            "n_error": 0,
            "n_cached_calls": 0,
            "n_failed_iterations": 0,
            "n_noop_candidates": 0,
        }
    )
    start_time: float = field(default_factory=time.perf_counter)
    compute_seconds: float = 0.0
    profile_text: str = ""
    run_id: str = ""
    iterations_used: int = 0
    stop_reason: str = ""
    memory_meta_features: list[float] = field(default_factory=list)
    memory_entries: list[dict[str, Any]] = field(default_factory=list)
    memory_context: str = ""


class WorkflowState(TypedDict):
    context: AgentState


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _short_error(error: str | None) -> str:
    if not error:
        return ""
    lines = [line.strip() for line in error.strip().splitlines() if line.strip()]
    return (lines[-1] if lines else "")[:300]


def _mark_iteration_failed(
    state: AgentState,
    trace_path: Path,
    iteration: int,
    reason: str,
) -> None:
    if not state.history:
        return
    final_attempt = state.history[-1]
    final_attempt["iteration_failed"] = True
    final_attempt["iteration_failure_reason"] = reason
    attempt_number = final_attempt["attempt"]

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in reversed(records):
        if (
            record.get("iteration") == iteration
            and record.get("attempt") == attempt_number
        ):
            record["iteration_failed"] = True
            record["iteration_failure_reason"] = reason
            break
    with trace_path.open("w", encoding="utf-8") as trace_file:
        for record in records:
            trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def profile(state: AgentState) -> None:
    X_train, y_train = load_train_only(state.dataset, state.seed)
    profile_data = profile_dataset(X_train, y_train)
    state.profile_text = profile_to_text(profile_data)
    state.memory_meta_features = meta_features(profile_data)


def propose(
    state: AgentState,
    client: LLMClient,
    run_id: str,
    *,
    base_seed: int,
    repair_error: str = "",
    repair_code: str = "",
    memory_context: str = "",
) -> tuple[LLMResult, str, str]:
    """Request one hypothesis and its corresponding build_model implementation."""
    recent_attempts = state.history[-5:]
    history_text = "\n".join(
        (
            f"- Hypothesis: {attempt['hypothesis']}; status: {attempt['status']}; "
            f"CV AUC: {attempt['cv_auc']}; delta: {attempt['delta']}; "
            f"reason: {attempt.get('error', '')}"
        )
        for attempt in recent_attempts
    ) or "(no previous candidate attempts)"
    optional_memory = (
        f"\nOptional memory context:\n{memory_context}\n" if memory_context else ""
    )
    repair_section = (
        f"\nThe previous candidate failed. Repair error: {repair_error}\n"
        f"Failing candidate code:\n```python\n{repair_code}\n```\n"
        "Keep the same idea as your previous attempt. Fix only the error shown. "
        "Do not replace it with the default XGBoost pipeline.\n"
        if repair_error
        else ""
    )
    user_prompt = (
        "Task: propose one sklearn-compatible binary classifier for ROC-AUC. "
        "Make ONE targeted change per iteration. Do not repeat a hypothesis "
        "already present in the attempt history. Never hardcode column names; "
        "derive columns from X_train (for example, with select_dtypes).\n\n"
        f"Dataset profile:\n{state.profile_text}\n\n"
        f"Rules and code template:\n{CODE_TEMPLATE}\n"
        f"{BUILD_MODEL_SKELETON}\n\n"
        "Output exactly a first line '# HYPOTHESIS: <one sentence>' followed "
        "by one ```python fenced block containing build_model(X_train, y_train). "
        "Do not put any other text before the hypothesis line.\n\n"
        f"Current best training-only CV AUC: {state.best_cv_auc}\n"
        f"Last five attempts:\n{history_text}\n"
        f"{repair_section}{optional_memory}"
    )
    call_seed = base_seed + state.counters["llm_calls"]
    try:
        result = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a careful tabular AutoML researcher. Follow the "
                        "required output format exactly."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            run_id=run_id,
            seed=call_seed,
        )
    finally:
        state.counters["llm_calls"] += 1

    state.counters["prompt_tokens"] += result.prompt_tokens
    state.counters["completion_tokens"] += result.completion_tokens
    state.counters["n_cached_calls"] += int(result.cached)
    first_line = result.text.splitlines()[0] if result.text.splitlines() else ""
    hypothesis_match = HYPOTHESIS_PATTERN.match(first_line)
    hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else ""
    return result, hypothesis, extract_code(result.text)


def validate(code: str, hypothesis: str) -> tuple[bool, str]:
    if not hypothesis:
        return False, "Response must begin with '# HYPOTHESIS: <one sentence>'."
    return validate_code(code)


def execute(code: str, state: AgentState) -> SandboxResult:
    return run_candidate(code, state.dataset, state.seed)


def evaluate(
    state: AgentState,
    code: str,
    result: SandboxResult,
) -> tuple[float | None, float | None]:
    previous_best = state.best_cv_auc
    cv_auc = result.cv_auc if result.status == "ok" else None
    delta = (
        float(cv_auc - previous_best)
        if cv_auc is not None and previous_best is not None
        else None
    )
    if cv_auc is not None and (
        previous_best is None or cv_auc > previous_best
    ):
        state.best_cv_auc = cv_auc
        state.best_code = code
    return cv_auc, delta


def _normalized_code(code: str) -> str:
    return " ".join(code.split())


def _is_noop_candidate(
    state: AgentState,
    code: str,
    cv_auc: float | None,
    incumbent_auc: float | None,
) -> bool:
    normalized_code = _normalized_code(code)
    repeated_code = any(
        _normalized_code(attempt["code"]) == normalized_code
        for attempt in state.history
    )
    same_score = (
        cv_auc is not None
        and incumbent_auc is not None
        and round(cv_auc, 6) == round(incumbent_auc, 6)
    )
    return repeated_code or same_score


def _log_attempt(
    state: AgentState,
    trace_path: Path,
    *,
    attempt: int,
    hypothesis: str,
    code: str,
    status: str,
    cv_auc: float | None,
    delta: float | None,
    error: str,
    prompt_tokens: int,
    completion_tokens: int,
    llm_seconds: float,
    sandbox_seconds: float,
    step: int,
    noop: bool,
) -> None:
    state.counters["attempts"] += 1
    if status in {"invalid_code", "timeout", "memory", "error"}:
        state.counters[f"n_{'invalid' if status == 'invalid_code' else status}"] += 1

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_record = {
        "iteration": step,
        "attempt": attempt,
        "hypothesis": hypothesis,
        "code": code,
        "status": status,
        "cv_auc": cv_auc,
        "delta": delta,
        "error": _short_error(error),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_seconds": llm_seconds,
        "sandbox_seconds": sandbox_seconds,
        "best_so_far": state.best_cv_auc,
        "noop": noop,
    }
    with trace_path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

    metrics = {
        "best_cv_auc": state.best_cv_auc,
        "sandbox_seconds": sandbox_seconds,
        "llm_seconds": llm_seconds,
    }
    if cv_auc is not None:
        metrics["candidate_cv_auc"] = cv_auc
    log_agent_iteration(state.run_id, step, metrics)


def _record_attempt(
    state: AgentState,
    trace_path: Path,
    *,
    attempt: int,
    hypothesis: str,
    code: str,
    status: str,
    cv_auc: float | None,
    delta: float | None,
    error: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    llm_seconds: float,
    sandbox_seconds: float,
    step: int,
    noop: bool = False,
) -> None:
    summary = _short_error(error)
    state.history.append(
        {
            "iteration": step,
            "attempt": attempt,
            "hypothesis": hypothesis,
            "status": status,
            "cv_auc": cv_auc,
            "delta": delta,
            "error": summary,
            "best_so_far": state.best_cv_auc,
            "code": code,
            "noop": noop,
        }
    )
    _log_attempt(
        state,
        trace_path,
        attempt=attempt,
        hypothesis=hypothesis,
        code=code,
        status=status,
        cv_auc=cv_auc,
        delta=delta,
        error=summary,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        llm_seconds=llm_seconds,
        sandbox_seconds=sandbox_seconds,
        step=step,
        noop=noop,
    )
    state.compute_seconds += llm_seconds + sandbox_seconds


def _record_infrastructure_error(
    state: AgentState,
    trace_path: Path,
    *,
    error: str,
    llm_seconds: float = 0.0,
    sandbox_seconds: float = 0.0,
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(
            json.dumps(
                {
                    "iteration": state.iteration,
                    "attempt": state.counters["attempts"],
                    "hypothesis": "",
                    "code": "",
                    "status": "infra_error",
                    "cv_auc": None,
                    "delta": None,
                    "error": _short_error(error),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "llm_seconds": llm_seconds,
                    "sandbox_seconds": sandbox_seconds,
                    "best_so_far": state.best_cv_auc,
                    "noop": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _default_code(seed: int) -> str:
    return DEFAULT_MODEL_CODE.replace("SEED_VALUE", str(seed))


def _print_table(history: list[dict[str, Any]]) -> None:
    print(f"{'iteration':>9}  {'hypothesis':<52}  {'status':<12}  {'CV AUC':>8}  {'best so far':>11}")
    print("-" * 105)
    for attempt in history:
        hypothesis = str(attempt["hypothesis"]).replace("\n", " ")
        if len(hypothesis) > 52:
            hypothesis = hypothesis[:49] + "..."
        score = attempt["cv_auc"]
        best = attempt.get("best_so_far")
        status = str(attempt["status"])
        if attempt.get("iteration_failed"):
            status = f"{status} (iteration failed)"
        print(
            f"{attempt['iteration']:>9}  {hypothesis:<52}  "
            f"{status:<25}  "
            f"{score if score is not None else '-':>8}  "
            f"{best if best is not None else '-':>11}"
        )


def run_agent(
    dataset: str,
    seed: int,
    *,
    client: LLMClient | None = None,
    results_file: Path | None = None,
    mode: str = "live",
    method: str = "plain",
    memory_db: Path | None = None,
    trace_dir: Path | None = None,
) -> dict[str, Any]:
    config = _load_config()
    agent_config = config["agent"]
    model_name = str(config["llm"]["model"])
    llm_seed = int(config["llm"].get("seed", 0))
    max_iterations = int(agent_config["max_iterations"])
    max_wall_seconds = float(agent_config["max_wall_seconds"])
    max_repairs = int(agent_config["max_repairs"])
    if mode not in {"mock", "live"}:
        raise ValueError(f"Unsupported agent mode: {mode!r}.")
    if method not in {"plain", "memory"}:
        raise ValueError(f"Unsupported agent method: {method!r}.")
    client = client or LLMClient()
    trace_root = trace_dir or (
        PROJECT_ROOT / "results" / "mock" / "traces"
        if mode == "mock"
        else PROJECT_ROOT / "results" / "traces"
    )
    trace_path = trace_root / f"{dataset}_{method}_{seed}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_header = {
        "type": "header",
        "mode": mode,
        "method": method,
        "memory_context": "",
    }
    trace_path.write_text(
        json.dumps(trace_header, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result_path = results_file or PROJECT_ROOT / "results" / (
        Path("mock") / "agent_results.csv" if mode == "mock" else Path("agent_results.csv")
    )
    state = AgentState(dataset=dataset, seed=seed, method=method)
    test_auc: float | None = None
    graph_values: dict[str, Any] = {
        "repairs_used": 0,
        "repair_error": "",
        "repair_code": "",
        "response": None,
        "hypothesis": "",
        "code": "",
        "llm_seconds": 0.0,
        "llm_error": "",
        "valid": False,
        "validation_error": "",
        "sandbox_result": None,
        "candidate_status": "",
        "candidate_error": "",
    }

    def note_infra(error: str, *, llm_seconds: float = 0.0, sandbox_seconds: float = 0.0) -> None:
        state.compute_seconds += llm_seconds + sandbox_seconds
        _record_infrastructure_error(
            state,
            trace_path,
            error=error,
            llm_seconds=llm_seconds,
            sandbox_seconds=sandbox_seconds,
        )

    def profile_node(_: WorkflowState) -> dict[str, Any]:
        for retry in range(3):
            started = time.perf_counter()
            try:
                profile(state)
                break
            except Exception as error:
                elapsed = time.perf_counter() - started
                if not is_infrastructure_error(error):
                    raise
                note_infra(str(error))
                if retry == 2:
                    state.stop_reason = "infra_error"
                    break
                time.sleep(0.5)
        if not state.stop_reason:
            if state.method == "memory":
                entries = retrieve(
                    state.dataset,
                    state.memory_meta_features,
                    db_path=memory_db or PROJECT_ROOT / "data" / "memory.db",
                )
                state.memory_entries = entries
                state.memory_context = format_memory_context(entries)
            trace_header["memory_context"] = state.memory_context
            trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
            trace_path.write_text(
                json.dumps(trace_header, ensure_ascii=False)
                + "\n"
                + "\n".join(trace_lines[1:])
                + ("\n" if len(trace_lines) > 1 else ""),
                encoding="utf-8",
            )
        return {}

    def baseline_node(_: WorkflowState) -> dict[str, Any]:
        if state.stop_reason:
            return {}
        baseline_code = _default_code(seed)
        baseline_result: SandboxResult | None = None
        for retry in range(3):
            started = time.perf_counter()
            try:
                baseline_result = execute(baseline_code, state)
            except Exception as error:
                elapsed = time.perf_counter() - started
                if not is_infrastructure_error(error):
                    raise
                note_infra(str(error), sandbox_seconds=elapsed)
                if retry == 2:
                    state.stop_reason = "infra_error"
                    return {}
                time.sleep(0.5)
                continue
            if baseline_result.status != "infra_error":
                break
            note_infra(
                baseline_result.error or "Infrastructure error in sandbox.",
                sandbox_seconds=baseline_result.seconds,
            )
            if retry == 2:
                state.stop_reason = "infra_error"
                return {}
            time.sleep(0.5)
        if baseline_result is None:
            state.stop_reason = "infra_error"
            return {}
        baseline_auc, baseline_delta = evaluate(state, baseline_code, baseline_result)
        _record_attempt(
            state,
            trace_path,
            attempt=0,
            hypothesis="Fixed default imputation, one-hot encoding, and XGBoost pipeline.",
            code=baseline_code,
            status=baseline_result.status,
            cv_auc=baseline_auc,
            delta=baseline_delta,
            error=baseline_result.error or "",
            llm_seconds=0.0,
            sandbox_seconds=baseline_result.seconds,
            step=0,
        )
        if baseline_result.status != "ok":
            state.stop_reason = "baseline_failed"
        else:
            state.iteration = 1
        return {}

    def propose_node(_: WorkflowState) -> dict[str, Any]:
        graph_values["response"] = None
        graph_values["hypothesis"] = ""
        graph_values["code"] = ""
        graph_values["llm_error"] = ""
        if state.stop_reason:
            graph_values["candidate_status"] = "infra_stop"
            return {}
        for retry in range(3):
            llm_started = time.perf_counter()
            try:
                response, hypothesis, code = propose(
                    state,
                    client,
                    state.run_id,
                    base_seed=llm_seed,
                    repair_error=graph_values["repair_error"],
                    repair_code=graph_values["repair_code"],
                    memory_context=state.memory_context,
                )
                graph_values["response"] = response
                graph_values["hypothesis"] = hypothesis
                graph_values["code"] = code
                graph_values["llm_seconds"] = response.seconds
                graph_values["candidate_status"] = ""
                break
            except Exception as exception:
                elapsed = time.perf_counter() - llm_started
                if not is_infrastructure_error(exception):
                    graph_values["llm_error"] = f"{type(exception).__name__}: {exception}"
                    graph_values["llm_seconds"] = elapsed
                    break
                note_infra(
                    f"{type(exception).__name__}: {exception}",
                    llm_seconds=elapsed,
                )
                if retry == 2:
                    state.stop_reason = "infra_error"
                    graph_values["candidate_status"] = "infra_stop"
                    break
                time.sleep(0.5)
        return {}

    def validate_node(_: WorkflowState) -> dict[str, Any]:
        if graph_values["candidate_status"] == "infra_stop":
            return {}
        if graph_values["llm_error"]:
            graph_values["valid"] = False
            graph_values["validation_error"] = graph_values["llm_error"]
        else:
            graph_values["valid"], graph_values["validation_error"] = validate(
                graph_values["code"], graph_values["hypothesis"]
            )
        return {}

    def execute_node(_: WorkflowState) -> dict[str, Any]:
        for retry in range(3):
            started = time.perf_counter()
            try:
                sandbox_result = execute(graph_values["code"], state)
            except Exception as error:
                elapsed = time.perf_counter() - started
                if not is_infrastructure_error(error):
                    raise
                note_infra(str(error), sandbox_seconds=elapsed)
                if retry == 2:
                    state.stop_reason = "infra_error"
                    graph_values["sandbox_result"] = SandboxResult(
                        "infra_error", None, str(error), elapsed, 0.0
                    )
                    break
                time.sleep(0.5)
                continue
            if sandbox_result.status != "infra_error":
                graph_values["sandbox_result"] = sandbox_result
                break
            note_infra(
                sandbox_result.error or "Infrastructure error in sandbox.",
                sandbox_seconds=sandbox_result.seconds,
            )
            if retry == 2:
                state.stop_reason = "infra_error"
                graph_values["sandbox_result"] = sandbox_result
                break
            time.sleep(0.5)
        return {}

    def evaluate_node(_: WorkflowState) -> dict[str, Any]:
        sandbox_result = graph_values["sandbox_result"]
        if (
            isinstance(sandbox_result, SandboxResult)
            and sandbox_result.status == "infra_error"
        ):
            state.stop_reason = "infra_error"
            graph_values["candidate_status"] = "stop"
            return {}
        response = graph_values["response"]
        prompt_tokens = response.prompt_tokens if response is not None else 0
        completion_tokens = response.completion_tokens if response is not None else 0
        llm_seconds = graph_values["llm_seconds"]
        noop = False
        if graph_values["llm_error"]:
            status = "error"
            cv_auc = None
            delta = None
            error = graph_values["llm_error"]
            sandbox_seconds = 0.0
        elif not graph_values["valid"]:
            status = "invalid_code"
            cv_auc = None
            delta = None
            error = graph_values["validation_error"]
            sandbox_seconds = 0.0
        else:
            if not isinstance(sandbox_result, SandboxResult):
                raise RuntimeError("Execute node did not return a sandbox result.")
            incumbent_auc = state.best_cv_auc
            cv_auc, delta = evaluate(state, graph_values["code"], sandbox_result)
            status = sandbox_result.status
            error = sandbox_result.error or ""
            sandbox_seconds = sandbox_result.seconds
            if status == "ok":
                noop = _is_noop_candidate(
                    state,
                    graph_values["code"],
                    cv_auc,
                    incumbent_auc,
                )
                if noop:
                    state.counters["n_noop_candidates"] += 1

        _record_attempt(
            state,
            trace_path,
            attempt=state.counters["attempts"],
            hypothesis=graph_values["hypothesis"],
            code=graph_values["code"],
            status=status,
            cv_auc=cv_auc,
            delta=delta,
            error=error,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_seconds=llm_seconds,
            sandbox_seconds=sandbox_seconds,
            step=state.iteration,
            noop=noop,
        )

        if status == "infra_error":
            state.stop_reason = "infra_error"
            graph_values["candidate_status"] = "stop"
            return {}

        now = state.compute_seconds
        if status != "ok":
            if graph_values["repairs_used"] < max_repairs:
                if now >= max_wall_seconds:
                    state.stop_reason = "max_wall_seconds"
                    graph_values["candidate_status"] = "stop"
                    return {}
                graph_values["repairs_used"] += 1
                graph_values["repair_error"] = _short_error(error) or (
                    f"Candidate status: {status}"
                )
                graph_values["repair_code"] = graph_values["code"]
                graph_values["candidate_status"] = "repair"
            else:
                state.iterations_used = state.iteration
                state.counters["n_failed_iterations"] += 1
                reason = _short_error(error) or f"Candidate status: {status}"
                _mark_iteration_failed(state, trace_path, state.iteration, reason)
                if now >= max_wall_seconds:
                    state.stop_reason = "max_wall_seconds"
                    graph_values["candidate_status"] = "stop"
                elif state.iteration >= max_iterations:
                    state.stop_reason = "max_iterations"
                    graph_values["candidate_status"] = "stop"
                else:
                    state.iteration += 1
                    graph_values["repairs_used"] = 0
                    graph_values["repair_error"] = ""
                    graph_values["repair_code"] = ""
                    graph_values["candidate_status"] = "continue"
        else:
            state.iterations_used = state.iteration
            graph_values["repairs_used"] = 0
            graph_values["repair_error"] = ""
            graph_values["repair_code"] = ""
            if now >= max_wall_seconds:
                state.stop_reason = "max_wall_seconds"
                graph_values["candidate_status"] = "stop"
            elif state.iteration >= max_iterations:
                state.stop_reason = "max_iterations"
                graph_values["candidate_status"] = "stop"
            else:
                state.iteration += 1
                graph_values["candidate_status"] = "continue"
        if state.stop_reason:
            graph_values["candidate_status"] = "stop"
        return {}

    def after_baseline(_: WorkflowState) -> str:
        if (
            not state.stop_reason
            and state.compute_seconds >= max_wall_seconds
        ):
            state.stop_reason = "max_wall_seconds"
        return "propose" if not state.stop_reason else END

    def after_validation(_: WorkflowState) -> str:
        if graph_values["candidate_status"] == "infra_stop":
            return END
        return "execute" if graph_values["valid"] else "evaluate"

    def after_evaluation(_: WorkflowState) -> str:
        return "propose" if graph_values["candidate_status"] in {"repair", "continue"} else END

    workflow = StateGraph(WorkflowState)
    workflow.add_node("profile", profile_node)
    workflow.add_node("baseline", baseline_node)
    workflow.add_node("propose", propose_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_edge(START, "profile")
    workflow.add_edge("profile", "baseline")
    workflow.add_conditional_edges("baseline", after_baseline)
    workflow.add_edge("propose", "validate")
    workflow.add_conditional_edges("validate", after_validation)
    workflow.add_edge("execute", "evaluate")
    workflow.add_conditional_edges("evaluate", after_evaluation)
    compiled_workflow = workflow.compile()

    with agent_run(dataset, method, seed) as run_id:
        state.run_id = run_id
        graph_values["repair_error"] = ""
        compiled_workflow.invoke({"context": state})
        if not state.stop_reason:
            state.stop_reason = "max_iterations"
        if state.best_code is not None:
            test_auc = final_test_auc(state.best_code, dataset, seed)

        log_agent_iteration(
            run_id,
            state.iteration,
            {},
            params={
                "stop_reason": state.stop_reason,
                "iterations_used": state.iterations_used,
                "llm_calls": state.counters["llm_calls"],
                "model": model_name,
            },
        )

    wall_seconds = time.perf_counter() - state.start_time
    _print_table(state.history)
    result = {
        "dataset": dataset,
        "method": method,
        "mode": mode,
        "seed": seed,
        "best_cv_auc": state.best_cv_auc,
        "test_auc": test_auc,
        "iterations_used": state.iterations_used,
        "llm_calls": state.counters["llm_calls"],
        "prompt_tokens": state.counters["prompt_tokens"],
        "completion_tokens": state.counters["completion_tokens"],
        "wall_seconds": wall_seconds,
        "compute_seconds": state.compute_seconds,
        "n_cached_calls": state.counters["n_cached_calls"],
        "n_invalid": state.counters["n_invalid"],
        "n_timeout": state.counters["n_timeout"],
        "n_memory": state.counters["n_memory"],
        "n_error": state.counters["n_error"],
        "n_failed_iterations": state.counters["n_failed_iterations"],
        "n_noop_candidates": state.counters["n_noop_candidates"],
        "n_memory_entries_used": len(state.memory_entries),
        "memory_datasets_used": list(
            dict.fromkeys(entry["dataset"] for entry in state.memory_entries)
        ),
        "memory_context_chars": len(state.memory_context),
        "stop_reason": state.stop_reason,
        "model": model_name,
    }
    _append_result(result_path, result)
    print(
        f"LLM calls={result['llm_calls']}, prompt_tokens={result['prompt_tokens']}, "
        f"completion_tokens={result['completion_tokens']}, "
        f"wall_seconds={wall_seconds:.2f}, "
        f"compute_seconds={state.compute_seconds:.2f}, "
        f"cached_calls={state.counters['n_cached_calls']}, final_test_auc={test_auc}, "
        f"failures(invalid/timeout/memory/error)="
        f"{result['n_invalid']}/{result['n_timeout']}/"
        f"{result['n_memory']}/{result['n_error']}, "
        f"noop_candidates={result['n_noop_candidates']}, "
        f"stop_reason={state.stop_reason}"
    )
    return result


def _append_result(path: Path, result: dict[str, Any]) -> None:
    import csv

    columns = (
        "dataset",
        "method",
        "mode",
        "seed",
        "best_cv_auc",
        "test_auc",
        "iterations_used",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "wall_seconds",
        "compute_seconds",
        "n_cached_calls",
        "n_invalid",
        "n_timeout",
        "n_memory",
        "n_error",
        "n_failed_iterations",
        "n_noop_candidates",
        "n_memory_entries_used",
        "memory_datasets_used",
        "memory_context_chars",
        "stop_reason",
        "model",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    rewrite = not path.exists() or path.stat().st_size == 0
    if not rewrite:
        with path.open(newline="", encoding="utf-8") as result_file:
            reader = csv.DictReader(result_file)
            existing_rows = list(reader)
            rewrite = reader.fieldnames != list(columns)
    if rewrite:
        for row in existing_rows:
            row.setdefault("mode", "live")
            row.setdefault("compute_seconds", "")
            row.setdefault("n_cached_calls", "")
            row.setdefault("n_failed_iterations", 0)
            row.setdefault("n_noop_candidates", 0)
            row.setdefault("n_memory_entries_used", 0)
            row.setdefault("memory_datasets_used", "[]")
            row.setdefault("memory_context_chars", 0)
        existing_rows.append(result)
        with path.open("w", newline="", encoding="utf-8") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        **row,
                        "memory_datasets_used": (
                            json.dumps(row["memory_datasets_used"])
                            if isinstance(row.get("memory_datasets_used"), list)
                            else row.get("memory_datasets_used", "[]")
                        ),
                    }
                    for row in existing_rows
                ]
            )
    else:
        with path.open("a", newline="", encoding="utf-8") as result_file:
            csv.DictWriter(result_file, fieldnames=columns).writerow(
                {
                    **result,
                    "memory_datasets_used": json.dumps(
                        result["memory_datasets_used"]
                    ),
                }
            )


def run_is_finished(
    path: Path,
    dataset: str,
    seed: int,
    method: str = "plain",
    mode: str = "live",
) -> bool:
    import csv

    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as result_file:
        reader = csv.DictReader(result_file)
        for row in reader:
            if (
                row.get("dataset") == dataset
                and row.get("method") == method
                and row.get("seed") == str(seed)
                and row.get("mode", "live") == mode
            ):
                return True
    return False
