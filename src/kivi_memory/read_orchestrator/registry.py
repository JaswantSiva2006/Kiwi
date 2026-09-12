"""Small read-tool registry used by the read router and response service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class SemanticMemorySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = None


class CalendarGetScheduleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str

    @model_validator(mode="after")
    def validate_interval(self) -> "CalendarGetScheduleArgs":
        start = _parse_datetime(self.start, "start")
        end = _parse_datetime(self.end, "end")
        if start >= end:
            raise ValueError("calendar start must be before end")
        if (
            start.hour == start.minute == start.second == start.microsecond == 0
            and end.date() == start.date()
            and end.hour == 23
            and end.minute == 59
        ):
            self.end = (datetime.combine(start.date(), datetime.min.time(), tzinfo=start.tzinfo) + timedelta(days=1)).isoformat()
        return self


class RedisThreadHistorySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = None
    start: str | None = None
    end: str | None = None
    limit: int = Field(default=8, ge=1, le=20)

    @model_validator(mode="after")
    def validate_optional_interval(self) -> "RedisThreadHistorySearchArgs":
        if self.start is None and self.end is None:
            return self
        if self.start is None or self.end is None:
            raise ValueError("redis history time filter requires both start and end")
        start = _parse_datetime(self.start, "start")
        end = _parse_datetime(self.end, "end")
        if start >= end:
            raise ValueError("redis history start must be before end")
        return self


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    freshness: str = "none"

    @model_validator(mode="after")
    def validate_web_search(self) -> "WebSearchArgs":
        if not self.query.strip():
            raise ValueError("web search query must not be empty")
        self.freshness = _normalize_freshness(self.freshness)
        if self.freshness not in {"none", "day", "week", "month", "year"}:
            raise ValueError("freshness must be one of none, day, week, month, year")
        if self.freshness == "none" and _text_needs_freshness(self.query):
            self.freshness = "week"
        return self


class MemoryControlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = None

    @model_validator(mode="after")
    def validate_memory_control(self) -> "MemoryControlArgs":
        if self.query is not None and not self.query.strip():
            raise ValueError("memory control query must not be empty")
        return self


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    execute: Callable[..., Any] | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()

    def validate_arguments(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        return self.input_model.model_validate(arguments or {}).model_dump(mode="json")


class ToolRegistry:
    """Registry for read-side tools; recent Redis context remains automatic."""

    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or default_tool_specs():
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if not spec.name.strip():
            raise ValueError("tool name must not be empty")
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool {spec.name!r}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ValueError(f"unknown tool {name!r}") from exc

    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def validate_calls(self, calls: list[Any]) -> list[dict[str, Any]]:
        validated = []
        seen: set[str] = set()
        for raw_call in calls:
            data = TypeAdapter(dict[str, Any]).validate_python(raw_call)
            tool = str(data.get("tool") or "")
            if tool in seen:
                raise ValueError(f"duplicate tool call {tool!r}")
            spec = self.get(tool)
            seen.add(tool)
            validated.append({"tool": tool, "arguments": spec.validate_arguments(data.get("arguments") or {})})
        return validated


def default_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="semantic_memory.search",
            description="Use for persistent facts, people, preferences, projects, goals, relationships and stored knowledge.",
            input_model=SemanticMemorySearchArgs,
        ),
        ToolSpec(
            name="calendar.get_schedule",
            description=(
                "Use for scheduled events, availability, deadlines and time-range calendar questions. "
                "Do not use calendar merely for historical/change questions such as what happened to, changed, moved, "
                "or old schedule; use semantic_memory.search for those unless the user also asks what is currently on the calendar. "
                "Resolve start/end from CURRENT_DATETIME and TIMEZONE as timezone-aware ISO-8601 using [start,end); "
                "whole-day ranges must end at the start of the following day."
            ),
            input_model=CalendarGetScheduleArgs,
        ),
        ToolSpec(
            name="redis_thread_history.search",
            description=(
                "Use for older same-thread conversation outside the supplied recent-context window. "
                "Use for phrases like mentioned earlier, much earlier, previous discussion, or old thread context. "
                "Arguments may include query and limit; include start/end only for an explicit history time range."
            ),
            input_model=RedisThreadHistorySearchArgs,
        ),
        ToolSpec(
            name="memory.control",
            description=(
                "Use when the user explicitly asks to inspect, explain, correct, remove, forget, "
                "or change Kivi's stored memory."
            ),
            input_model=MemoryControlArgs,
        ),
        ToolSpec(
            name="web.search",
            description=(
                "Use only when answering requires current or external internet information not reliably available "
                "from the supplied conversation or Kivi's stored information, including latest/current/recent news, "
                "opportunities, events, availability, public facts or status. Provide a concise search query and "
                "appropriate freshness."
            ),
            input_model=WebSearchArgs,
        ),
    ]


def _parse_datetime(value: str, name: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


def _text_needs_freshness(text: str) -> bool:
    lowered = text.casefold()
    return any(word in lowered for word in ["latest", "current", "today", "recent", "news", "opportunities", "events", "status"])


def _normalize_freshness(value: str) -> str:
    normalized = str(value or "none").strip().casefold()
    return {
        "fresh": "week",
        "recent": "week",
        "current": "week",
        "latest": "week",
        "high": "week",
        "low": "month",
    }.get(normalized, normalized)
