"""Ollama-backed JSON judge for non-obvious reconciliation cases."""

from __future__ import annotations

import json
import time
from typing import Any

from kivi_memory.reconciliation.config import (
    RECONCILIATION_MODEL,
    RECONCILIATION_NUM_PREDICT,
    RECONCILIATION_OLLAMA_BASE_URL,
    RECONCILIATION_OLLAMA_TIMEOUT_SECONDS,
    RECONCILIATION_TEMPERATURE,
    RECONCILIATION_THINK,
)
from kivi_memory.reconciliation.models import JudgeResult
from kivi_memory.reconciliation.prompt import RECONCILIATION_SYSTEM_PROMPT
from kivi_memory.reconciliation.validator import MODEL_DECISION_SCHEMA
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaResponseError


class OllamaReconciliationJudge:
    """Small wrapper around Ollama chat that preserves token metrics."""

    def __init__(self, client: OllamaClient | None = None, model: str = RECONCILIATION_MODEL) -> None:
        self.client = client or OllamaClient(
            base_url=RECONCILIATION_OLLAMA_BASE_URL,
            timeout_seconds=RECONCILIATION_OLLAMA_TIMEOUT_SECONDS,
        )
        self.model = model

    def decide(self, compact_input: str, retry_note: str | None = None) -> JudgeResult:
        user_content = compact_input
        if retry_note:
            user_content = f"{compact_input}\n\nPrevious output was invalid: {retry_note}\nReturn valid JSON only."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": RECONCILIATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": MODEL_DECISION_SCHEMA,
            "options": {
                "temperature": RECONCILIATION_TEMPERATURE,
                "num_predict": RECONCILIATION_NUM_PREDICT,
            },
            "think": RECONCILIATION_THINK,
        }
        started = time.perf_counter()
        response = self.client._post("/api/chat", payload)
        llm_ms = (time.perf_counter() - started) * 1000
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama response did not contain message.content")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Ollama returned malformed generated JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise OllamaResponseError("Ollama generated JSON was not an object")
        return JudgeResult(
            raw_decision=raw,
            llm_ms=llm_ms,
            prompt_eval_count=_int_or_none(response.get("prompt_eval_count")),
            eval_count=_int_or_none(response.get("eval_count")),
        )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
