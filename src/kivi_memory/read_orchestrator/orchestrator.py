"""Read-side router for choosing registered read tools."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from kivi_memory.common.config import KiviOrchestratorConfig, load_orchestrator_config_from_env
from kivi_memory.read_orchestrator.models import ReadOrchestrationResult, ReadToolDecision, RouterToolCall
from kivi_memory.read_orchestrator.prompt import READ_ROUTER_SYSTEM_PROMPT, build_read_router_system_prompt
from kivi_memory.read_orchestrator.registry import ToolRegistry
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError, OllamaResponseError
from kivi_memory.working_memory.models import ThreadEpisode, ThreadMessage

READ_ROUTER_THINK = False
READ_ROUTER_MAX_ATTEMPTS = 2


class ReadSideOrchestrator:
    """One-call read router; it does not fetch sources or answer the user."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        config: KiviOrchestratorConfig | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config or load_orchestrator_config_from_env()
        self.registry = registry or ToolRegistry()
        self.client = client or OllamaClient(
            base_url=self.config.ollama_base_url,
            timeout_seconds=self.config.timeout_seconds,
        )

    def route(
        self,
        current_query: str,
        recent_thread_context: list[ThreadEpisode | ThreadMessage],
        *,
        current_datetime: datetime,
        user_timezone: str,
        document_probe: dict[str, Any] | None = None,
    ) -> ReadOrchestrationResult:
        if not current_query.strip():
            raise ValueError("current_query must not be empty")
        if current_datetime.tzinfo is None or current_datetime.utcoffset() is None:
            raise ValueError("current_datetime must be timezone-aware")
        if not user_timezone.strip():
            raise ValueError("user_timezone must not be empty")

        user_content = format_read_router_input(
            current_query=current_query,
            recent_thread_context=recent_thread_context,
            current_datetime=current_datetime,
            user_timezone=user_timezone,
            document_probe=document_probe,
        )
        started = time.perf_counter()
        calls = 0
        prompt_eval_count = None
        eval_count = None
        last_error: Exception | None = None

        for attempt in range(1, READ_ROUTER_MAX_ATTEMPTS + 1):
            calls += 1
            content = user_content
            if attempt > 1:
                content = f"{user_content}\n\nPrevious output was invalid. Return only valid JSON matching the schema."
            try:
                raw_response = self._call_model(content)
                prompt_eval_count = raw_response.prompt_eval_count
                eval_count = raw_response.eval_count
                decision = self._validate_decision(raw_response.raw, current_query, current_datetime, user_timezone)
                return ReadOrchestrationResult(
                    decision=decision,
                    router_llm_ms=_elapsed_ms(started),
                    router_model_call_count=calls,
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                )
            except (OllamaClientError, ValidationError, ValueError) as exc:
                last_error = exc

        fallback = ReadToolDecision(tool_calls=[])
        return ReadOrchestrationResult(
            decision=fallback,
            router_llm_ms=_elapsed_ms(started),
            router_model_call_count=calls,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            fallback_used=True,
            error=str(last_error) if last_error else "read router failed",
        )

    def _call_model(self, user_content: str) -> "_RawReadRouterResponse":
        system_prompt = build_read_router_system_prompt(_format_tool_descriptions(self.registry))
        payload = {
            "model": self.config.router_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": ReadToolDecision.model_json_schema(),
            "options": {"temperature": self.config.router_temperature},
            "think": READ_ROUTER_THINK,
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
        return _RawReadRouterResponse(
            raw=parsed,
            prompt_eval_count=_int_or_none(response.get("prompt_eval_count")),
            eval_count=_int_or_none(response.get("eval_count")),
        )

    def _validate_decision(
        self,
        raw: dict[str, Any],
        current_query: str,
        current_datetime: datetime,
        user_timezone: str,
    ) -> ReadToolDecision:
        decision = ReadToolDecision.model_validate(raw)
        raw_calls = [
            _repair_tool_arguments(call.model_dump(mode="json"), current_query, current_datetime, user_timezone)
            for call in decision.tool_calls
        ]
        raw_calls = _add_missing_obvious_web_call(raw_calls, current_query)
        validated_calls = self.registry.validate_calls(raw_calls)
        return ReadToolDecision(
            tool_calls=[
                RouterToolCall(tool=call["tool"], arguments=call["arguments"])
                for call in validated_calls
            ]
        )


@dataclass(frozen=True)
class _RawReadRouterResponse:
    raw: dict[str, Any]
    prompt_eval_count: int | None
    eval_count: int | None


def format_read_router_input(
    *,
    current_query: str,
    recent_thread_context: list[ThreadEpisode | ThreadMessage],
    current_datetime: datetime,
    user_timezone: str,
    document_probe: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"CURRENT_DATETIME: {current_datetime.isoformat()}",
        f"TIMEZONE: {user_timezone}",
        "",
        "CURRENT THREAD:",
    ]
    if recent_thread_context:
        for role, text in _flatten_thread_context(recent_thread_context):
            lines.append(f"{role}: {_one_line(text)}")
    else:
        lines.append("(none)")
    if document_probe is not None:
        lines.extend(
            [
                "",
                "DOCUMENT_RELEVANCE_PROBE:",
                json.dumps(document_probe, ensure_ascii=False, sort_keys=True),
            ]
        )
    lines.extend(["", "CURRENT USER QUERY:", _one_line(current_query)])
    return "\n".join(lines)


def _format_tool_descriptions(registry: ToolRegistry) -> str:
    return "\n".join(f"- {spec.name}: {spec.description}" for spec in registry.specs())


def _flatten_thread_context(messages: list[ThreadEpisode | ThreadMessage]) -> list[tuple[str, str]]:
    flattened = []
    for item in messages:
        if isinstance(item, ThreadEpisode):
            flattened.extend((message.role, message.text) for message in item.messages)
        else:
            flattened.append((item.role, item.text))
    return flattened


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _repair_tool_arguments(
    call: dict[str, Any],
    current_query: str,
    current_datetime: datetime,
    user_timezone: str,
) -> dict[str, Any]:
    if call.get("tool") == "web.search":
        return _repair_web_arguments(call, current_query)
    if call.get("tool") != "calendar.get_schedule":
        return call
    target_weekday = _explicit_next_weekday(current_query)
    if target_weekday is None and "next week" not in current_query.casefold():
        return call
    arguments = dict(call.get("arguments") or {})
    start_raw = arguments.get("start")
    end_raw = arguments.get("end")
    if not isinstance(start_raw, str) or not isinstance(end_raw, str):
        return call
    try:
        start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
    except ValueError:
        return call
    zoned_now = current_datetime.astimezone(ZoneInfo(user_timezone))
    if target_weekday is None:
        target_date = _next_week_start_date(zoned_now)
        repaired_start = datetime.combine(target_date, datetime.min.time(), tzinfo=zoned_now.tzinfo)
        repaired_end = repaired_start + timedelta(days=7)
        arguments["start"] = repaired_start.isoformat()
        arguments["end"] = repaired_end.isoformat()
        return {**call, "arguments": arguments}
    target_date = _next_weekday_date(zoned_now, target_weekday)
    if start.date() == target_date:
        return call
    duration = end - start
    repaired_start = datetime.combine(target_date, start.timetz())
    repaired_end = repaired_start + duration
    arguments["start"] = repaired_start.isoformat()
    arguments["end"] = repaired_end.isoformat()
    return {**call, "arguments": arguments}


def _repair_web_arguments(call: dict[str, Any], current_query: str) -> dict[str, Any]:
    arguments = dict(call.get("arguments") or {})
    freshness = arguments.get("freshness")
    if freshness == "none" and _query_needs_freshness(current_query):
        arguments["freshness"] = "week"
    return {**call, "arguments": arguments}


def _query_needs_freshness(text: str) -> bool:
    lowered = text.casefold()
    return any(word in lowered for word in ["latest", "current", "today", "recent", "news", "opportunities", "events"])


def _add_missing_obvious_web_call(calls: list[dict[str, Any]], current_query: str) -> list[dict[str, Any]]:
    calls = _add_missing_obvious_memory_control_call(calls, current_query)
    if any(call.get("tool") == "web.search" for call in calls):
        return calls
    if not _query_obviously_needs_web(current_query):
        return calls
    return [
        *calls,
        {
            "tool": "web.search",
            "arguments": {
                "query": _one_line(current_query),
                "freshness": "week",
            },
        },
    ]


def _add_missing_obvious_memory_control_call(calls: list[dict[str, Any]], current_query: str) -> list[dict[str, Any]]:
    if any(call.get("tool") == "memory.control" for call in calls):
        return calls
    lowered = current_query.casefold()
    if not any(phrase in lowered for phrase in ["what do you remember", "why did you think", "forget", "delete that", "remove that", "doesn't", "does not", "anymore"]):
        return calls
    return [
        *calls,
        {
            "tool": "memory.control",
            "arguments": {"query": _one_line(current_query)},
        },
    ]


def _query_obviously_needs_web(text: str) -> bool:
    lowered = text.casefold()
    return (
        any(word in lowered for word in ["latest", "news", "opportunities"])
        or "current status" in lowered
        or ("find" in lowered and any(word in lowered for word in ["events", "attend", "opportunities"]))
    )


def _explicit_next_weekday(text: str) -> int | None:
    lowered = text.casefold()
    for name, index in _WEEKDAYS.items():
        if f"next {name}" in lowered:
            return index
    return None


def _next_weekday_date(current_datetime: datetime, target_weekday: int):
    days_ahead = (target_weekday - current_datetime.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (current_datetime + timedelta(days=days_ahead)).date()


def _next_week_start_date(current_datetime: datetime):
    days_until_next_monday = (7 - current_datetime.weekday()) % 7
    if days_until_next_monday == 0:
        days_until_next_monday = 7
    return (current_datetime + timedelta(days=days_until_next_monday)).date()
