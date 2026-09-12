"""Ollama-backed router for deciding whether long-term memory is needed."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kivi_memory.common.config import (
    KiviOrchestratorConfig,
    load_orchestrator_config_from_env,
)
from kivi_memory.llm_redis_orchestrator.prompt import ROUTER_SYSTEM_PROMPT
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError, OllamaResponseError
from kivi_memory.working_memory.models import ThreadMessage

ROUTER_THINK = False
ROUTER_MAX_ATTEMPTS = 2


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouterOutput(_StrictBaseModel):
    needs_long_term_memory: bool
    retrieval_queries: list[str] = Field(default_factory=list)

    @field_validator("retrieval_queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        cleaned = [" ".join(query.split()) for query in queries]
        if any(not query for query in cleaned):
            raise ValueError("retrieval_queries must contain non-empty strings")
        return cleaned

    @model_validator(mode="after")
    def validate_consistency(self) -> "RouterOutput":
        if not self.needs_long_term_memory and self.retrieval_queries:
            raise ValueError("retrieval_queries must be empty when needs_long_term_memory is false")
        if self.needs_long_term_memory and not self.retrieval_queries:
            raise ValueError("retrieval_queries must be present when needs_long_term_memory is true")
        return self


@dataclass(frozen=True)
class RouterCallResult:
    output: RouterOutput
    router_llm_ms: float
    router_model_call_count: int
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    fallback_used: bool = False
    error: str | None = None


class LongTermMemoryRouter:
    """One-call classifier/query generator for answer-time memory preparation."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        config: KiviOrchestratorConfig | None = None,
    ) -> None:
        self.config = config or load_orchestrator_config_from_env()
        self.client = client or OllamaClient(
            base_url=self.config.ollama_base_url,
            timeout_seconds=self.config.timeout_seconds,
        )

    def route(self, prior_messages: list[ThreadMessage], current_query: str) -> RouterCallResult:
        if not current_query.strip():
            raise ValueError("current_query must not be empty")

        user_content = format_router_input(prior_messages, current_query)
        started = time.perf_counter()
        calls = 0
        prompt_eval_count = None
        eval_count = None
        last_error: Exception | None = None

        for attempt in range(1, ROUTER_MAX_ATTEMPTS + 1):
            calls += 1
            content = user_content
            if attempt > 1:
                content = f"{user_content}\n\nPrevious output was invalid. Return only valid JSON matching the schema."
            try:
                raw_response = self._call_model(content)
                prompt_eval_count = raw_response.prompt_eval_count
                eval_count = raw_response.eval_count
                output = RouterOutput.model_validate(raw_response.raw)
                output = _enforce_query_limit(output, self.config.router_max_retrieval_queries)
                output = _apply_obvious_user_memory_guard(output, current_query)
                return RouterCallResult(
                    output=output,
                    router_llm_ms=_elapsed_ms(started),
                    router_model_call_count=calls,
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                )
            except (OllamaClientError, ValidationError, ValueError) as exc:
                last_error = exc

        fallback = RouterOutput(needs_long_term_memory=True, retrieval_queries=[_one_line(current_query)])
        return RouterCallResult(
            output=fallback,
            router_llm_ms=_elapsed_ms(started),
            router_model_call_count=calls,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            fallback_used=True,
            error=str(last_error) if last_error else "router failed",
        )

    def _call_model(self, user_content: str) -> "_RawRouterResponse":
        payload = {
            "model": self.config.router_model,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": RouterOutput.model_json_schema(),
            "options": {"temperature": self.config.router_temperature},
            "think": ROUTER_THINK,
        }
        response = self.client._post("/api/chat", payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama response did not contain message.content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Ollama returned malformed generated JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise OllamaResponseError("Ollama generated JSON was not an object")
        return _RawRouterResponse(
            raw=parsed,
            prompt_eval_count=_int_or_none(response.get("prompt_eval_count")),
            eval_count=_int_or_none(response.get("eval_count")),
        )


@dataclass(frozen=True)
class _RawRouterResponse:
    raw: dict[str, Any]
    prompt_eval_count: int | None
    eval_count: int | None


def format_router_input(prior_messages: list[ThreadMessage], current_query: str) -> str:
    lines = ["PRIOR THREAD CONTEXT:"]
    if prior_messages:
        for message in prior_messages:
            lines.append(f"{message.role}: {_one_line(message.text)}")
    else:
        lines.append("(none)")
    lines.extend(["", "CURRENT USER QUERY:", _one_line(current_query)])
    return "\n".join(lines)


def _enforce_query_limit(output: RouterOutput, max_queries: int) -> RouterOutput:
    if output.needs_long_term_memory and len(output.retrieval_queries) > max_queries:
        raise ValueError(f"retrieval_queries must contain at most {max_queries} queries")
    return output


def _apply_obvious_user_memory_guard(output: RouterOutput, current_query: str) -> RouterOutput:
    if output.needs_long_term_memory or not _obvious_user_memory_question(current_query):
        return output
    return RouterOutput(needs_long_term_memory=True, retrieval_queries=[_one_line(current_query)])


def _obvious_user_memory_question(current_query: str) -> bool:
    query = current_query.lower()
    if not re.search(r"\b(i|me|my|mine|user)\b", query):
        return False
    return bool(
        re.search(
            r"\b(prefer|preference|like|liked|habit|usually|typically|history|remember|decide|decided|"
            r"work(?:ed|ing)?|handle(?:d|s)?|own(?:ed|s)?|meet|met|plan(?:ned|s|ning)?|used to|no longer)\b",
            query,
        )
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
