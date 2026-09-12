"""Deterministic calendar projection from enriched semantic assertions."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kivi_memory.calendar.repository import CalendarRepository
from kivi_memory.common.schemas import CalendarEventExtraction, CandidateSemanticAssertion, MemoryEpisode, MemoryType, Polarity, Recurrence, TemporalMetadata, TemporalPrecision


def create_calendar_event_for_memory(
    cur,
    *,
    memory_id: str,
    assertion: CandidateSemanticAssertion | dict[str, Any],
    temporal_metadata: TemporalMetadata | dict[str, Any] | None,
    entity_resolution: dict[str, Any],
    episode: MemoryEpisode | dict[str, Any] | None = None,
    repository: CalendarRepository | None = None,
) -> str | None:
    """Create one calendar row for a positive calendar semantic memory."""

    assertion = CandidateSemanticAssertion.model_validate(_plain(assertion))
    temporal = TemporalMetadata.model_validate(_plain(temporal_metadata)) if temporal_metadata else None
    if not _should_create_scheduled_event(assertion, temporal):
        return None

    repo = repository or CalendarRepository()
    return repo.insert_calendar_event_in_transaction(
        cur,
        event_values=_event_values(memory_id, assertion, temporal, episode),
        entity_links=_entity_links(assertion, entity_resolution),
    )


def mark_calendar_event_superseded(
    cur,
    *,
    memory_id: str,
    repository: CalendarRepository | None = None,
) -> int:
    return (repository or CalendarRepository()).update_calendar_status_for_memory_in_transaction(
        cur,
        memory_id=memory_id,
        status="SUPERSEDED",
    )


def cancel_calendar_event_for_memory(
    cur,
    *,
    memory_id: str,
    repository: CalendarRepository | None = None,
) -> int:
    return (repository or CalendarRepository()).update_calendar_status_for_memory_in_transaction(
        cur,
        memory_id=memory_id,
        status="CANCELLED",
    )


def _should_create_scheduled_event(
    assertion: CandidateSemanticAssertion,
    temporal: TemporalMetadata | None,
) -> bool:
    if assertion.memory_type != MemoryType.CALENDAR_EVENT:
        return False
    if assertion.polarity != Polarity.POSITIVE:
        return False
    if _calendar_payload(assertion, temporal) is None:
        return False
    calendar = _calendar_payload(assertion, temporal)
    if temporal is None:
        return bool(calendar and (calendar.start_time_text or calendar.recurrence_text))
    return bool(
        temporal.event_time
        or temporal.valid_from_hint
        or temporal.valid_to_hint
        or _enum_value(temporal.recurrence) != Recurrence.NONE.value
        or calendar.start_time_text
        or calendar.recurrence_text
    )


def _event_values(
    memory_id: str,
    assertion: CandidateSemanticAssertion,
    temporal: TemporalMetadata | None,
    episode: MemoryEpisode | dict[str, Any] | None,
) -> dict[str, Any]:
    calendar = _calendar_payload(assertion, temporal)
    if calendar is None:
        raise ValueError("calendar_event is required")

    start_at, start_date = _temporal_point(temporal.event_time if temporal else None, temporal)
    valid_from_at, valid_from_date = _temporal_point(temporal.valid_from_hint if temporal else None, temporal)
    end_at, end_date = _temporal_point(temporal.valid_to_hint if temporal else None, temporal)
    if start_at is None and start_date is None:
        start_at, start_date = valid_from_at, valid_from_date
    if _enum_value(temporal.recurrence) != Recurrence.NONE.value if temporal else False:
        recurrence_until_at, recurrence_until_date = end_at, end_date
        described_start_at, described_start_date = _recurrence_anchor(assertion, episode, temporal, calendar)
        if described_start_at is not None or described_start_date is not None:
            start_at, start_date = described_start_at, described_start_date
        end_at, end_date = _occurrence_end_from_calendar(start_at, start_date, calendar)
        if start_at is None and start_date is None:
            start_at, start_date = _recurrence_anchor(assertion, episode, temporal, calendar)
    else:
        recurrence_until_at, recurrence_until_date = None, None
        if start_at is None and start_date is None:
            start_at, start_date = _described_schedule_anchor(assertion, episode, calendar)

    return {
        "semantic_memory_id": memory_id,
        "title": calendar.title,
        "event_kind": calendar.event_kind,
        "start_at": start_at,
        "end_at": end_at,
        "start_date": start_date,
        "end_date": end_date,
        "recurrence_until_at": recurrence_until_at,
        "recurrence_until_date": recurrence_until_date,
        "all_day": bool(calendar.all_day_hint),
        "location_text": calendar.location_text,
        "timezone_text": calendar.timezone_text,
        "recurrence": _enum_value(temporal.recurrence) if temporal else None,
        "recurrence_specifics": temporal.recurrence_specifics if temporal else None,
        "status": "SCHEDULED",
    }


def _recurrence_anchor(
    assertion: CandidateSemanticAssertion,
    episode: MemoryEpisode | dict[str, Any] | None,
    temporal: TemporalMetadata | None,
    calendar,
) -> tuple[str | None, str | None]:
    """Anchor recurrence expansion to the described schedule, not observation time."""

    reference = _first_observed_datetime(assertion, episode)
    if reference is None:
        return None, None
    timezone_text = calendar.timezone_text or (episode.timezone if isinstance(episode, MemoryEpisode) else None)
    display_tz = _timezone(timezone_text) or reference.tzinfo
    if display_tz is not None and reference.tzinfo is not None:
        reference = reference.astimezone(display_tz)

    source = " ".join(
        part
        for part in (
            calendar.start_time_text,
            calendar.recurrence_text,
            temporal.recurrence_specifics if temporal else None,
            assertion.canonical_text,
        )
        if part
    )
    return _anchor_from_text(source, reference)


def _described_schedule_anchor(
    assertion: CandidateSemanticAssertion,
    episode: MemoryEpisode | dict[str, Any] | None,
    calendar,
) -> tuple[str | None, str | None]:
    reference = _first_observed_datetime(assertion, episode)
    if reference is None:
        return None, None
    timezone_text = calendar.timezone_text or (episode.timezone if isinstance(episode, MemoryEpisode) else None)
    display_tz = _timezone(timezone_text) or reference.tzinfo
    if display_tz is not None and reference.tzinfo is not None:
        reference = reference.astimezone(display_tz)
    source = " ".join(part for part in (calendar.start_time_text, assertion.canonical_text) if part)
    return _anchor_from_text(source, reference)


def _anchor_from_text(source: str, reference: datetime) -> tuple[str | None, str | None]:
    weekday = _weekday_from_text(source)
    if weekday is None:
        return None, None

    anchor_date = reference.date() + timedelta(days=(weekday - reference.weekday()) % 7)
    clock = _clock_from_text(source)
    if clock is None:
        daypart_hour = _daypart_hour(source)
        if daypart_hour is None:
            return None, anchor_date.isoformat()
        clock = (daypart_hour, 0)
    anchor = reference.replace(
        year=anchor_date.year,
        month=anchor_date.month,
        day=anchor_date.day,
        hour=clock[0],
        minute=clock[1],
        second=0,
        microsecond=0,
    )
    if anchor < reference:
        anchor += timedelta(days=7)
    return anchor.isoformat(), None


def _occurrence_end_from_calendar(
    start_at: str | None,
    start_date: str | None,
    calendar,
) -> tuple[str | None, str | None]:
    if calendar.end_time_text and start_at:
        parsed = _parse_time_on_anchor(start_at, calendar.end_time_text)
        if parsed:
            return parsed, None
    if calendar.duration_text and start_at:
        parsed = _duration_end(start_at, calendar.duration_text)
        if parsed:
            return parsed, None
    if start_date and calendar.all_day_hint:
        return None, start_date
    return None, None


def _calendar_payload(assertion: CandidateSemanticAssertion, temporal: TemporalMetadata | None):
    if temporal is not None and temporal.calendar_event is not None:
        return temporal.calendar_event
    if assertion.calendar_event is not None:
        return assertion.calendar_event
    if assertion.memory_type == MemoryType.CALENDAR_EVENT:
        return _derive_calendar_payload(assertion)
    return None


def _derive_calendar_payload(assertion: CandidateSemanticAssertion) -> CalendarEventExtraction:
    text = assertion.canonical_text.strip()
    schedule_parts = [
        argument.text
        for argument in assertion.semantic_arguments
        if not argument.is_entity and _looks_schedule_text(argument.role, argument.text)
    ]
    schedule_text = " ".join(schedule_parts) or text
    return CalendarEventExtraction(
        title=_derive_title(text),
        event_kind=_derive_event_kind(text),
        location_text=_argument_text(assertion, {"location", "place", "venue"}),
        start_time_text=schedule_text,
        end_time_text=None,
        duration_text=_argument_text(assertion, {"duration"}),
        recurrence_text=schedule_text if _looks_recurring_text(schedule_text) else None,
        timezone_text=None,
        all_day_hint=False if _clock_from_text(schedule_text) else None,
    )


def _derive_title(text: str) -> str:
    title = re.sub(r"^\b(the|a|an|user's|the user's)\b\s+", "", text, flags=re.IGNORECASE).strip()
    title = re.sub(r"\b(is|are|was|were|scheduled|set|usually|every|on|at|this|next)\b.*$", "", title, flags=re.IGNORECASE).strip()
    return title[:1].upper() + title[1:] if title else "Calendar event"


def _derive_event_kind(text: str) -> str:
    lowered = text.casefold()
    if any(token in lowered for token in ("review", "meeting", "sync", "check-in", "check in")):
        return "MEETING"
    if any(token in lowered for token in ("deadline", "due")):
        return "DEADLINE"
    if "appointment" in lowered:
        return "APPOINTMENT"
    return "OTHER"


def _argument_text(assertion: CandidateSemanticAssertion, roles: set[str]) -> str | None:
    for argument in assertion.semantic_arguments:
        if argument.role.casefold() in roles and argument.text.strip():
            return argument.text
    return None


def _looks_schedule_text(role: str, text: str) -> bool:
    value = f"{role} {text}".casefold()
    return any(
        token in value
        for token in (
            "time",
            "date",
            "schedule",
            "frequency",
            "recurrence",
            "every",
            "usually",
            "this week",
            "today",
            "tomorrow",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        )
    )


def _looks_recurring_text(text: str) -> bool:
    return any(token in text.casefold() for token in ("every", "usually", "daily", "weekly", "monthly", "yearly"))


def _first_observed_datetime(
    assertion: CandidateSemanticAssertion,
    episode: MemoryEpisode | dict[str, Any] | None,
) -> datetime | None:
    if episode is None:
        return None
    if not isinstance(episode, MemoryEpisode):
        episode = MemoryEpisode.model_validate(_plain(episode))
    message_ids = {span.message_id for span in assertion.source_spans}
    observed = [message.timestamp for message in episode.messages if message.message_id in message_ids]
    if not observed:
        return None
    timestamp = observed[0]
    if isinstance(timestamp, datetime):
        return timestamp
    return _parse_datetime(str(timestamp))


def _observed_anchor(
    assertion: CandidateSemanticAssertion,
    episode: MemoryEpisode | dict[str, Any] | None,
    temporal: TemporalMetadata | None,
) -> tuple[str | None, str | None]:
    timestamp = _first_observed_datetime(assertion, episode)
    if timestamp is None:
        return None, None
    value = timestamp.isoformat()
    start_at, start_date = _temporal_point(value, temporal)
    if start_at is None and start_date is None:
        return value, None
    return start_at, start_date


def _weekday_from_text(text: str) -> int | None:
    lowered = text.casefold()
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, index in weekdays.items():
        if re.search(rf"\b{name}s?\b", lowered):
            return index
    return None


def _clock_from_text(text: str) -> tuple[int, int] | None:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", text, flags=re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        suffix = match.group(3).lower()
        if suffix.startswith("p") and hour != 12:
            hour += 12
        if suffix.startswith("a") and hour == 12:
            hour = 0
        return hour, minute
    match = re.search(r"(?<![+-])\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _daypart_hour(text: str) -> int | None:
    lowered = text.casefold()
    if re.search(r"\bmorning\b", lowered):
        return 9
    if re.search(r"\bafternoon\b", lowered):
        return 14
    if re.search(r"\bevening\b", lowered):
        return 18
    if re.search(r"\bnight\b", lowered):
        return 20
    return None


def _timezone(value: str | None):
    if not value:
        return None
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError:
        return None


def _parse_time_on_anchor(anchor: str, text: str) -> str | None:
    parsed_anchor = _parse_datetime(anchor)
    if parsed_anchor is None:
        return None
    cleaned = text.strip().upper().replace(".", "")
    for suffix in (" AM", " PM"):
        if cleaned.endswith(suffix):
            hour_text = cleaned[: -len(suffix)].strip().split(":")
            try:
                hour = int(hour_text[0])
                minute = int(hour_text[1]) if len(hour_text) > 1 else 0
            except ValueError:
                return None
            if suffix.strip() == "PM" and hour != 12:
                hour += 12
            if suffix.strip() == "AM" and hour == 12:
                hour = 0
            return parsed_anchor.replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat()
    return None


def _duration_end(anchor: str, text: str) -> str | None:
    parsed_anchor = _parse_datetime(anchor)
    if parsed_anchor is None:
        return None
    words = text.strip().casefold().split()
    for index, word in enumerate(words[:-1]):
        try:
            amount = float(word)
        except ValueError:
            continue
        unit = words[index + 1]
        if unit.startswith("hour"):
            return (parsed_anchor + timedelta(hours=amount)).isoformat()
        if unit.startswith("minute"):
            return (parsed_anchor + timedelta(minutes=amount)).isoformat()
    return None


def _temporal_point(value: str | None, temporal: TemporalMetadata | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    precision = _enum_value(temporal.temporal_precision) if temporal else None
    if precision == TemporalPrecision.DAY.value and "T" not in value:
        return None, value[:10]
    parsed = _parse_datetime(value)
    if parsed and parsed.time().isoformat() == "00:00:00" and precision == TemporalPrecision.DAY.value:
        return None, parsed.date().isoformat()
    if "T" in value:
        return value, None
    if len(value) == 10:
        return None, value
    return value, None


def _entity_links(assertion: CandidateSemanticAssertion, entity_resolution: dict[str, Any]) -> list[dict[str, Any]]:
    links = []
    subject_id = _resolved_entity_id(entity_resolution.get("subject"))
    if subject_id:
        links.append({"entity_id": subject_id, "role": "SUBJECT", "mention_text": assertion.subject.text})

    resolved_arguments = iter(entity_resolution.get("semantic_arguments") or [])
    for argument in assertion.semantic_arguments:
        argument_resolution = next(resolved_arguments, {}).get("result") if argument.is_entity else None
        entity_id = _resolved_entity_id(argument_resolution)
        if entity_id:
            links.append(
                {
                    "entity_id": entity_id,
                    "role": _calendar_role(argument.role),
                    "mention_text": argument.text,
                }
            )
    return links


def _calendar_role(argument_role: str) -> str:
    cleaned = "_".join(argument_role.upper().split())
    return cleaned or "RELATED"


def _resolved_entity_id(resolution: dict[str, Any] | None) -> str | None:
    if not resolution or resolution.get("resolution") not in {"MATCHED", "NEW"}:
        return None
    return resolution.get("entity_id")


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value
