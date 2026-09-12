"""Structured schedule result models for the read-only Calendar Tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CalendarOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calendar_event_id: str
    semantic_memory_id: str
    memory_created_at: str | None = None
    title: str
    event_kind: str | None = None
    start: str | None = None
    end: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    all_day: bool = False
    location_text: str | None = None
    recurring: bool = False
    recurrence: str | None = None
    recurrence_specifics: str | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class ScheduleDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_candidate_query_ms: float
    recurrence_expansion_ms: float
    total_ms: float
    db_recurring_series_considered: int
    occurrences_returned: int
    incomplete_recurring_rows: list[str] = Field(default_factory=list)


class ScheduleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[CalendarOccurrence]
    diagnostics: ScheduleDiagnostics
