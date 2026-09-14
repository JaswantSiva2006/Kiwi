"""Deterministic context assembly for read-side answer prompts."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
import re
from typing import Any

from pydantic import BaseModel

from kivi_memory.read_orchestrator.models import ReadOrchestrationResult, ReadSourceDecision, ReadToolDecision, ToolExecutionResult
from kivi_memory.working_memory.models import ThreadEpisode, ThreadMessage

MAX_CALENDAR_CONTEXT_EVENTS = 12
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "do",
    "for",
    "have",
    "i",
    "is",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "there",
    "to",
    "was",
    "what",
    "when",
    "with",
}


def build_context(
    query: str,
    thread_context: list[ThreadEpisode | ThreadMessage | dict[str, Any]],
    semantic_memories: list[Any] | None = None,
    calendar_result: Any | None = None,
    tool_results: list[ToolExecutionResult | dict[str, Any]] | None = None,
    orchestrator_result: ReadOrchestrationResult | ReadSourceDecision | dict[str, Any] | None = None,
) -> str:
    """Build the deterministic prompt context for the selected read route."""

    if not query.strip():
        raise ValueError("query must not be empty")

    normalized_results = _normalize_tool_results(tool_results)
    sections = [
        "<CURRENT_THREAD>",
        _format_thread(thread_context),
        "</CURRENT_THREAD>",
    ]
    if normalized_results:
        for tool, payload in normalized_results:
            sections.extend(["", *_format_tool_section(tool, payload, query=query)])
    else:
        use_semantic, use_calendar = _route_flags(orchestrator_result, semantic_memories, calendar_result)
        if use_semantic:
            sections.extend(
                [
                    "",
                    "<SEMANTIC_MEMORY>",
                    _format_semantic_memories(semantic_memories or []),
                    "</SEMANTIC_MEMORY>",
                ]
            )
        if use_calendar:
            sections.extend(
                [
                    "",
                    "<CALENDAR>",
                    _format_calendar_result(calendar_result, query=query),
                    "</CALENDAR>",
                ]
            )
    sections.extend(
        [
            "",
            "<USER_QUERY>",
            _one_line(query),
            "</USER_QUERY>",
        ]
    )
    return "\n".join(sections)


def _format_tool_section(tool: str, payload: Any, *, query: str) -> list[str]:
    if tool == "semantic_memory.search":
        return ["<SEMANTIC_MEMORY>", _format_semantic_memories(payload or []), "</SEMANTIC_MEMORY>"]
    if tool == "calendar.get_schedule":
        return ["<CALENDAR>", _format_calendar_result(payload, query=query), "</CALENDAR>"]
    if tool == "redis_thread_history.search":
        return ["<REDIS_THREAD_HISTORY>", _format_redis_history(payload), "</REDIS_THREAD_HISTORY>"]
    if tool == "web.search":
        return ["<WEB_CONTEXT>", _format_web_context(payload), "</WEB_CONTEXT>"]
    if tool == "document.search":
        return ["<DOCUMENT_CONTEXT>", _format_document_context(payload), "</DOCUMENT_CONTEXT>"]
    if tool == "memory.control":
        return ["<MEMORY_CONTROL>", _format_memory_control(payload), "</MEMORY_CONTROL>"]
    tag = tool.upper().replace(".", "_")
    return [f"<{tag}>", json.dumps(_to_plain(payload), ensure_ascii=False, sort_keys=True), f"</{tag}>"]


def _normalize_tool_results(
    tool_results: list[ToolExecutionResult | dict[str, Any]] | None,
) -> list[tuple[str, Any]]:
    normalized = []
    for item in tool_results or []:
        if isinstance(item, ToolExecutionResult):
            normalized.append((item.tool, item.result))
        elif isinstance(item, dict):
            normalized.append((str(item.get("tool") or ""), item.get("result")))
    return [(tool, result) for tool, result in normalized if tool]


def _route_flags(
    orchestrator_result: ReadOrchestrationResult | ReadSourceDecision | dict[str, Any] | None,
    semantic_memories: list[Any] | None,
    calendar_result: Any | None,
) -> tuple[bool, bool]:
    if isinstance(orchestrator_result, ReadOrchestrationResult):
        decision = orchestrator_result.decision
        return _decision_has_tool(decision, "semantic_memory.search"), _decision_has_tool(decision, "calendar.get_schedule")
    if isinstance(orchestrator_result, ReadToolDecision):
        return _decision_has_tool(orchestrator_result, "semantic_memory.search"), _decision_has_tool(orchestrator_result, "calendar.get_schedule")
    if isinstance(orchestrator_result, ReadSourceDecision):
        return orchestrator_result.use_semantic_memory, orchestrator_result.use_calendar
    if isinstance(orchestrator_result, dict):
        decision = orchestrator_result.get("decision", orchestrator_result)
        return bool(decision.get("use_semantic_memory")), bool(decision.get("use_calendar"))
    return bool(semantic_memories), calendar_result is not None


def _decision_has_tool(decision: ReadToolDecision, tool: str) -> bool:
    return any(call.tool == tool for call in decision.tool_calls)


def _format_thread(messages: list[ThreadEpisode | ThreadMessage | dict[str, Any]]) -> str:
    flattened = _flatten_thread_context(messages)
    if not flattened:
        return "(none)"
    return "\n".join(f"{role}: {_one_line(text)}" for role, text in flattened)


def _flatten_thread_context(messages: list[ThreadEpisode | ThreadMessage | dict[str, Any]]) -> list[tuple[str, str]]:
    flattened = []
    for item in messages:
        if isinstance(item, ThreadEpisode):
            flattened.extend((message.role, message.text) for message in item.messages)
        elif isinstance(item, dict) and isinstance(item.get("messages"), list):
            for message in item["messages"]:
                flattened.append((_message_field(message, "role"), _message_field(message, "text")))
        else:
            flattened.append((_message_field(item, "role"), _message_field(item, "text")))
    return flattened


def _format_semantic_memories(memories: list[Any]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(_format_semantic_memory(memory, index) for index, memory in enumerate(memories, start=1))


def _format_semantic_memory(memory: Any, index: int) -> str:
    data = _to_plain(memory)
    return json.dumps(
        {
            "label": f"M{index}",
            "memory_id": data.get("memory_id"),
            "canonical_text": data.get("canonical_text"),
            "created_at": data.get("created_at"),
            "subject": data.get("subject"),
            "predicate_type": data.get("predicate_type"),
            "arguments": data.get("arguments", []),
            "temporal": data.get("temporal"),
            "evidence": data.get("evidence", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _format_calendar_result(calendar_result: Any | None, *, query: str = "") -> str:
    if calendar_result is None:
        return "(none)"
    data = _to_plain(calendar_result)
    events = data.get("events") if isinstance(data, dict) else data
    if not events:
        return "[]"
    selected = _select_calendar_context_events(events, query=query)
    return json.dumps(
        {
            "events": selected,
            "returned_events": len(selected),
            "total_events": len(events),
            "selection": "query_ranked_bounded",
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _select_calendar_context_events(events: list[Any], *, query: str) -> list[dict[str, Any]]:
    plain_events = [_to_plain(event) for event in events]
    deduped = _dedupe_recurring_calendar_events(plain_events)
    tokens = _query_tokens(query)
    ranked = sorted(
        deduped,
        key=lambda event: (
            _calendar_relevance_score(event, tokens),
            _event_recency_timestamp(event),
        ),
        reverse=True,
    )
    return [_compact_calendar_event(event) for event in ranked[:MAX_CALENDAR_CONTEXT_EVENTS]]


def _compact_calendar_event(event: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for item in event.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        evidence.append(
            {
                "source_text": item.get("source_text"),
                "observed_at": item.get("observed_at"),
                "episode_id": item.get("episode_id"),
            }
        )
    participants = []
    for entity in event.get("entities") or []:
        if isinstance(entity, dict) and entity.get("mention_text"):
            participants.append({"text": entity.get("mention_text"), "role": entity.get("role")})
    return {
        "calendar_event_id": event.get("calendar_event_id"),
        "semantic_memory_id": event.get("semantic_memory_id"),
        "memory_created_at": event.get("memory_created_at"),
        "title": event.get("title"),
        "event_kind": event.get("event_kind"),
        "start": event.get("start"),
        "end": event.get("end"),
        "start_date": event.get("start_date"),
        "end_date": event.get("end_date"),
        "all_day": event.get("all_day"),
        "location_text": event.get("location_text"),
        "recurring": event.get("recurring"),
        "recurrence": event.get("recurrence"),
        "recurrence_specifics": _clip_text(event.get("recurrence_specifics"), 64),
        "participants": participants[:6],
        "evidence": evidence[:2],
    }


def _dedupe_recurring_calendar_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[str, dict[str, Any]] = {}
    singles: list[dict[str, Any]] = []
    for event in events:
        key = None
        if event.get("recurring"):
            key = str(event.get("semantic_memory_id") or event.get("calendar_event_id") or "")
        if not key:
            singles.append(event)
            continue
        current = by_series.get(key)
        if current is None or _event_recency_timestamp(event) > _event_recency_timestamp(current):
            by_series[key] = event
    return [*singles, *by_series.values()]


def _calendar_relevance_score(event: dict[str, Any], tokens: set[str]) -> int:
    if not tokens:
        return 0
    haystack = _calendar_search_text(event)
    score = sum(1 for token in tokens if token in haystack)
    title = str(event.get("title") or "").casefold()
    evidence = " ".join(str(item.get("source_text") or "") for item in event.get("evidence") or []).casefold()
    score += sum(2 for token in tokens if token in title)
    score += sum(2 for token in tokens if token in evidence)
    return score


def _calendar_search_text(event: dict[str, Any]) -> str:
    parts = [
        event.get("title"),
        event.get("event_kind"),
        event.get("location_text"),
        event.get("recurrence"),
        event.get("recurrence_specifics"),
        event.get("semantic_memory_id"),
        event.get("calendar_event_id"),
    ]
    for entity in event.get("entities") or []:
        if isinstance(entity, dict):
            parts.extend([entity.get("mention_text"), entity.get("role")])
    for evidence in event.get("evidence") or []:
        if isinstance(evidence, dict):
            parts.extend([evidence.get("source_text"), evidence.get("episode_id"), evidence.get("observed_at")])
    return " ".join(str(part).casefold() for part in parts if part)


def _clip_text(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _query_tokens(query: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", query.casefold()) if len(token) > 2 and token not in _QUERY_STOPWORDS}


def _event_recency_timestamp(event: dict[str, Any]) -> float:
    candidates = (
        event.get("memory_created_at"),
        event.get("start"),
        event.get("start_date"),
    )
    for value in candidates:
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed.timestamp()
    return 0.0


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        text = str(value)
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            text = f"{text}T00:00:00+00:00"
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_redis_history(history_result: Any | None) -> str:
    if history_result is None:
        return "(none)"
    data = _to_plain(history_result)
    episodes = data.get("episodes") if isinstance(data, dict) else data
    if not episodes:
        return "[]"
    return json.dumps(episodes, ensure_ascii=False, sort_keys=True)


def _format_web_context(web_result: Any | None) -> str:
    if web_result is None:
        return "(none)"
    data = _to_plain(web_result)
    sources = data.get("sources") if isinstance(data, dict) else data
    if not sources:
        return "[]"
    blocks = []
    for source in sources:
        item = _to_plain(source)
        blocks.append(
            "\n".join(
                [
                    f"[{item.get('source_id')}]",
                    f"Title: {item.get('title')}",
                    f"Date: {item.get('published_at') or 'unknown'}",
                    f"URL: {item.get('url')}",
                    f"Relevant content: {_one_line(item.get('relevant_text') or '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_document_context(document_result: Any | None) -> str:
    if document_result is None:
        return "(none)"
    data = _to_plain(document_result)
    results = data.get("results") if isinstance(data, dict) else []
    if not results:
        return "[]"
    blocks = []
    for result in results:
        item = _to_plain(result)
        section_path = item.get("section_path") or []
        section = item.get("section_title") or (" > ".join(section_path) if section_path else "unknown")
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        pages = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        blocks.append(
            "\n".join(
                [
                    f"[{item.get('source_id')}]",
                    f"File: {item.get('filename')}",
                    f"Section: {section}",
                    f"Pages: {pages}",
                    "Text:",
                    str(item.get("text") or "").strip(),
                ]
            )
        )
    return "\n\n".join(blocks)


def _format_memory_control(memory_control_result: Any | None) -> str:
    if memory_control_result is None:
        return "(none)"
    data = _to_plain(memory_control_result)
    return json.dumps(
        {
            "action": data.get("action"),
            "status": data.get("status"),
            "message": data.get("message"),
            "affected_memory_ids": data.get("affected_memory_ids", []),
            "memories": data.get("memories", []),
            "provenance": data.get("provenance", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_plain(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _message_field(message: ThreadMessage | dict[str, Any], field: str) -> str:
    if isinstance(message, dict):
        return str(message.get(field) or "")
    return str(getattr(message, field))


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())
