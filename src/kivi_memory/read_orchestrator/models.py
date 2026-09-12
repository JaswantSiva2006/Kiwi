"""Data models for read-side tool routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RouterToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReadToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[RouterToolCall] = Field(default_factory=list)


class ReadSourceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_semantic_memory: bool
    use_calendar: bool
    calendar_start: datetime | None
    calendar_end: datetime | None

    @model_validator(mode="after")
    def validate_calendar_interval(self) -> "ReadSourceDecision":
        if self.use_calendar:
            if self.calendar_start is None or self.calendar_end is None:
                raise ValueError("use_calendar=true requires calendar_start and calendar_end")
            if self.calendar_start.tzinfo is None or self.calendar_start.utcoffset() is None:
                raise ValueError("calendar_start must be timezone-aware")
            if self.calendar_end.tzinfo is None or self.calendar_end.utcoffset() is None:
                raise ValueError("calendar_end must be timezone-aware")
            if self.calendar_start >= self.calendar_end:
                raise ValueError("calendar_start must be before calendar_end")
        elif self.calendar_start is not None or self.calendar_end is not None:
            raise ValueError("calendar_start and calendar_end must be null when use_calendar=false")
        return self


@dataclass(frozen=True)
class ReadOrchestrationResult:
    decision: ReadToolDecision
    router_llm_ms: float
    router_model_call_count: int
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    fallback_used: bool = False
    error: str | None = None

    @property
    def use_semantic_memory(self) -> bool:
        return any(call.tool == "semantic_memory.search" for call in self.decision.tool_calls)

    @property
    def use_calendar(self) -> bool:
        return any(call.tool == "calendar.get_schedule" for call in self.decision.tool_calls)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool: str
    arguments: dict[str, Any]
    result: Any
    latency_ms: float


@dataclass(frozen=True)
class ToolExecutionPlan:
    calls: list[RouterToolCall] = field(default_factory=list)
