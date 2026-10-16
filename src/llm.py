"""Local LLM client with a persistent response cache and SQLite call log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cached: bool


class ChatProvider(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> ProviderResponse: ...


class OllamaProvider:
    """Ollama implementation of the small chat-provider interface."""

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> ProviderResponse:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "seed": seed,
                    },
                },
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as error:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Start Ollama and ensure its local API is available."
            ) from error
        except requests.exceptions.Timeout as error:
            raise RuntimeError(
                f"Ollama at {self.base_url} did not respond within "
                f"{self.timeout:g} seconds."
            ) from error

        if not response.ok:
            detail = response.text.strip().replace("\n", " ")[:500]
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} for "
                f"{self.model!r}: {detail}"
            )

        payload = response.json()
        try:
            text = payload["message"]["content"]
            prompt_tokens = int(payload["prompt_eval_count"])
            completion_tokens = int(payload["eval_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Ollama returned an incomplete chat response; expected message "
                "content, prompt_eval_count, and eval_count."
            ) from error
        if not isinstance(text, str):
            raise RuntimeError("Ollama returned non-text message content.")
        return ProviderResponse(text, prompt_tokens, completion_tokens)


def _load_settings() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    settings = config.get("llm", {})
    provider_name = settings.get("provider", "ollama")
    if provider_name != "ollama":
        raise ValueError(f"Unsupported LLM provider configured: {provider_name!r}.")
    return {
        "model": str(settings.get("model", "qwen2.5-coder:7b")),
        "base_url": str(settings.get("base_url", "http://localhost:11434")),
        "timeout": float(settings.get("timeout", 600)),
        "seed": int(settings.get("seed", 0)),
    }


class LLMClient:
    def __init__(
        self,
        *,
        _provider: ChatProvider | None = None,
        _cache_dir: Path | None = None,
        _log_db_path: Path | None = None,
    ) -> None:
        settings = _load_settings()
        self.model = settings["model"]
        self.seed = settings["seed"]
        self._provider = _provider or OllamaProvider(
            settings["base_url"], self.model, settings["timeout"]
        )
        self._cache_dir = _cache_dir or PROJECT_ROOT / "data" / "llm_cache"
        self._log_db_path = _log_db_path or PROJECT_ROOT / "data" / "llm_log.db"

    def _cache_key(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        cache_input = json.dumps(
            [self.model, messages, temperature, max_tokens, seed],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()

    def _log_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        seconds: float,
        cached: bool,
        run_id: str | None,
    ) -> None:
        self._log_db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._log_db_path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_calls (
                        timestamp TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        seconds REAL NOT NULL,
                        cached INTEGER NOT NULL,
                        run_id TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO llm_calls (
                        timestamp, model, prompt_tokens, completion_tokens,
                        seconds, cached, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        self.model,
                        prompt_tokens,
                        completion_tokens,
                        seconds,
                        int(cached),
                        run_id,
                    ),
                )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
        run_id: str | None = None,
        seed: int | None = None,
    ) -> LLMResult:
        call_seed = self.seed if seed is None else int(seed)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / (
            f"{self._cache_key(messages, temperature, max_tokens, call_seed)}.json"
        )
        if cache_file.exists():
            cached_result = json.loads(cache_file.read_text(encoding="utf-8"))
            result = LLMResult(
                text=str(cached_result["text"]),
                prompt_tokens=int(cached_result["prompt_tokens"]),
                completion_tokens=int(cached_result["completion_tokens"]),
                seconds=float(cached_result["seconds"]),
                cached=True,
            )
            self._log_call(
                result.prompt_tokens,
                result.completion_tokens,
                result.seconds,
                True,
                run_id,
            )
            return result

        started = time.perf_counter()
        try:
            provider_result = self._provider.chat(
                messages, temperature, max_tokens, call_seed
            )
        except Exception:
            self._log_call(0, 0, time.perf_counter() - started, False, run_id)
            raise
        seconds = time.perf_counter() - started
        result = LLMResult(
            provider_result.text,
            provider_result.prompt_tokens,
            provider_result.completion_tokens,
            seconds,
            False,
        )
        self._log_call(
            result.prompt_tokens,
            result.completion_tokens,
            result.seconds,
            False,
            run_id,
        )
        cache_record = {
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "seconds": result.seconds,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._cache_dir,
            suffix=".tmp",
            delete=False,
        ) as cache_temp:
            json.dump(cache_record, cache_temp, ensure_ascii=False)
            temp_path = Path(cache_temp.name)
        temp_path.replace(cache_file)
        return result
