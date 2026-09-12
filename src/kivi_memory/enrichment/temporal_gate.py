"""Batched routing gate for temporal processing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kivi_memory.common.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_TIMEOUT_SECONDS
from kivi_memory.common.schemas import CandidateSemanticAssertion
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError, OllamaResponseError

TEMPORAL_GATE_MODEL = "qwen3.5:9b"
TEMPORAL_GATE_THINK = False
TEMPORAL_GATE_TEMPERATURE = 0
TEMPORAL_GATE_MAX_ATTEMPTS = 2

TEMPORAL_GATE_SYSTEM_PROMPT = """You are a routing gate for temporal processing of semantic memories.

For each memory, choose exactly one route:

NONE:
The memory has no temporal meaning that affects when the fact/event/state is
true or valid. No temporal processing is needed.

REASON:
The memory has temporal meaning, but no absolute or relative calendar/date/time
value needs to be normalized. Use this for temporal state or recurrence meaning
such as currently, still, no longer, used to, every week, every month, every
July, etc., when there is no explicit start/end/date/time anchor.

BOTH:
The memory contains temporal information that requires normalized date/time or
interval values. Use this for explicit or relative dates/times, clock times,
start/end boundaries, deadlines, schedules, expiry dates, approximate dated
events, or relative expressions such as yesterday, tomorrow, next week, since
2024, until March, from X to Y, around November 2026, or starting next month.

Rules:

- Temporal words matter only when they describe when the memory is true/event
  occurs. Do not classify something as temporal merely because a date/month word
  is part of the content itself.
- Recurrence without an absolute start/end anchor normally uses REASON.
- Current-state/change markers such as currently, still, no longer, anymore,
  previously, and used to normally use REASON unless they also contain a
  date/time anchor.
- If normalized temporal values are needed, choose BOTH. Never choose
  normalization without reasoning.
- If uncertain between NONE and a temporal route, prefer the temporal route.
- If uncertain between REASON and BOTH, choose BOTH.

Return only the required JSON. No explanation."""


class TemporalRoute(StrEnum):
    NONE = "NONE"
    REASON = "REASON"
    BOTH = "BOTH"


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalRouteItem(_StrictBaseModel):
    id: str
    route: TemporalRoute


class TemporalGateOutput(_StrictBaseModel):
    routes: list[TemporalRouteItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_routes(self) -> "TemporalGateOutput":
        if not self.routes:
            raise ValueError("routes must not be empty")
        return self


@dataclass(frozen=True)
class TemporalGateResult:
    routes: dict[str, TemporalRoute]
    temporal_gate_ms: float
    gate_prompt_eval_count: int | None
    gate_eval_count: int | None
    gate_model_call_count: int
    fallback_used: bool = False
    error: str | None = None


class TemporalRoutingGate:
    """Route all valid assertions in one model call."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        model: str = TEMPORAL_GATE_MODEL,
    ) -> None:
        self.client = client or OllamaClient(
            base_url=DEFAULT_OLLAMA_BASE_URL,
            timeout_seconds=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        )
        self.model = model

    def route(self, assertions: list[CandidateSemanticAssertion]) -> TemporalGateResult:
        if not assertions:
            return TemporalGateResult({}, 0.0, None, None, 0)

        assertion_ids = [f"A{index + 1}" for index in range(len(assertions))]
        user_content = format_temporal_gate_input(assertion_ids, assertions)
        started = time.perf_counter()
        calls = 0
        prompt_eval_count = None
        eval_count = None
        last_error: Exception | None = None

        for attempt in range(1, TEMPORAL_GATE_MAX_ATTEMPTS + 1):
            calls += 1
            content = user_content
            if attempt > 1:
                content = f"{user_content}\n\nPrevious output was invalid. Return valid JSON for every supplied ID."
            try:
                raw_response = self._call_model(content)
                prompt_eval_count = raw_response.prompt_eval_count
                eval_count = raw_response.eval_count
                output = TemporalGateOutput.model_validate(raw_response.raw)
                routes = validate_gate_routes(output, assertion_ids)
                return TemporalGateResult(
                    routes=routes,
                    temporal_gate_ms=(time.perf_counter() - started) * 1000,
                    gate_prompt_eval_count=prompt_eval_count,
                    gate_eval_count=eval_count,
                    gate_model_call_count=calls,
                )
            except (OllamaClientError, ValidationError, ValueError) as exc:
                last_error = exc

        return TemporalGateResult(
            routes={assertion_id: TemporalRoute.BOTH for assertion_id in assertion_ids},
            temporal_gate_ms=(time.perf_counter() - started) * 1000,
            gate_prompt_eval_count=prompt_eval_count,
            gate_eval_count=eval_count,
            gate_model_call_count=calls,
            fallback_used=True,
            error=str(last_error) if last_error else "temporal gate failed",
        )

    def _call_model(self, user_content: str) -> "_RawGateResponse":
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": TEMPORAL_GATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": TemporalGateOutput.model_json_schema(),
            "options": {"temperature": TEMPORAL_GATE_TEMPERATURE},
            "think": TEMPORAL_GATE_THINK,
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
        return _RawGateResponse(
            raw=parsed,
            prompt_eval_count=_int_or_none(response.get("prompt_eval_count")),
            eval_count=_int_or_none(response.get("eval_count")),
        )


@dataclass(frozen=True)
class _RawGateResponse:
    raw: dict[str, Any]
    prompt_eval_count: int | None
    eval_count: int | None


def route_temporal_assertions(
    assertions: list[CandidateSemanticAssertion],
    gate: TemporalRoutingGate | None = None,
) -> TemporalGateResult:
    return (gate or TemporalRoutingGate()).route(assertions)


def validate_gate_routes(output: TemporalGateOutput, assertion_ids: list[str]) -> dict[str, TemporalRoute]:
    expected = set(assertion_ids)
    seen = []
    routes = {}
    for route in output.routes:
        if route.id not in expected:
            raise ValueError(f"temporal gate returned invented ID: {route.id}")
        if route.id in routes:
            raise ValueError(f"temporal gate returned duplicate ID: {route.id}")
        seen.append(route.id)
        routes[route.id] = route.route

    if len(seen) != len(assertion_ids):
        raise ValueError("temporal gate output count does not match input count")
    missing = expected - set(seen)
    if missing:
        raise ValueError(f"temporal gate missing ID(s): {sorted(missing)}")
    return routes


def format_temporal_gate_input(
    assertion_ids: list[str],
    assertions: list[CandidateSemanticAssertion],
) -> str:
    return "\n".join(
        f"{assertion_id}|{_one_line(assertion.canonical_text)}"
        for assertion_id, assertion in zip(assertion_ids, assertions, strict=True)
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
