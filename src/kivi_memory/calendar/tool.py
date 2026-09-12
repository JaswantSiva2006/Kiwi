"""Read-only schedule retrieval over calendar projection rows."""

from __future__ import annotations

import time
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil.rrule import DAILY, MONTHLY, WEEKLY, YEARLY, rrule

from kivi_memory.calendar.models import CalendarOccurrence, ScheduleDiagnostics, ScheduleResult
from kivi_memory.calendar.repository import CalendarRepository


class CalendarTool:
    """Expand persisted calendar projection rows into requested occurrences."""

    def __init__(self, repository: CalendarRepository | None = None) -> None:
        self.repository = repository or CalendarRepository()

    def get_schedule(self, start: datetime, end: datetime) -> ScheduleResult:
        _validate_interval(start, end)
        total_started = time.perf_counter()
        db_started = time.perf_counter()
        rows = self.repository.fetch_schedule_candidates(start=start, end=end)
        db_ms = _elapsed_ms(db_started)

        expansion_started = time.perf_counter()
        events: list[CalendarOccurrence] = []
        incomplete_recurring_rows: list[str] = []
        recurring_count = 0
        for row in rows:
            if _is_recurring(row):
                recurring_count += 1
                expanded = _expand_recurring(row, start, end)
                if expanded is None:
                    incomplete_recurring_rows.append(str(row.get("calendar_event_id")))
                    continue
                events.extend(expanded)
                continue

            occurrence = _one_time_occurrence(row, start, end)
            if occurrence is not None:
                events.append(occurrence)

        events.sort(key=_sort_key)
        expansion_ms = _elapsed_ms(expansion_started)
        return ScheduleResult(
            events=events,
            diagnostics=ScheduleDiagnostics(
                db_candidate_query_ms=db_ms,
                recurrence_expansion_ms=expansion_ms,
                total_ms=_elapsed_ms(total_started),
                db_recurring_series_considered=recurring_count,
                occurrences_returned=len(events),
                incomplete_recurring_rows=incomplete_recurring_rows,
            ),
        )


def get_schedule(start: datetime, end: datetime) -> ScheduleResult:
    return CalendarTool().get_schedule(start, end)


def _one_time_occurrence(row: dict[str, Any], query_start: datetime, query_end: datetime) -> CalendarOccurrence | None:
    display_tz = _event_timezone(row, query_start.tzinfo)
    start_at = _aware(row.get("start_at"), query_start)
    if start_at is not None:
        end_at = _aware(row.get("end_at"), start_at) or start_at
        if not (start_at < query_end and end_at >= query_start):
            return None
        return _occurrence(row, start=start_at.astimezone(display_tz).isoformat(), end=end_at.astimezone(display_tz).isoformat())

    start_date = _date(row.get("start_date"))
    if start_date is None:
        return None
    end_date = _date(row.get("end_date")) or start_date
    if not _date_range_overlaps(start_date, end_date, query_start, query_end):
        return None
    return _occurrence(row, start_date=start_date.isoformat(), end_date=end_date.isoformat())


def _expand_recurring(row: dict[str, Any], query_start: datetime, query_end: datetime) -> list[CalendarOccurrence] | None:
    start_at = _aware(row.get("start_at"), query_start)
    start_date = _date(row.get("start_date"))
    recurrence = row.get("recurrence")
    freq = _rrule_frequency(recurrence)
    if freq is None:
        return None
    if start_at is not None:
        return _expand_timed_recurring(row, start_at, query_start, query_end, freq)
    if start_date is not None:
        return _expand_date_recurring(row, start_date, query_start, query_end, freq)
    return None


def _expand_timed_recurring(
    row: dict[str, Any],
    anchor: datetime,
    query_start: datetime,
    query_end: datetime,
    freq: int,
) -> list[CalendarOccurrence]:
    event_tz = _event_timezone(row, query_start.tzinfo)
    anchor = anchor.astimezone(event_tz)
    local_query_start = query_start.astimezone(event_tz)
    local_query_end = query_end.astimezone(event_tz)
    duration = _duration(anchor, _aware(row.get("end_at"), anchor))
    until = _recurrence_until_datetime(row, event_tz)
    before = min(local_query_end, until) if until is not None else local_query_end
    after = local_query_start - (duration or timedelta())
    rule = rrule(freq, dtstart=anchor, **_rrule_options(row, anchor, row.get("recurrence")))
    events = []
    for occurrence_start in rule.between(after, before, inc=True):
        occurrence_end = occurrence_start + duration if duration is not None else occurrence_start
        if occurrence_start < local_query_end and occurrence_end >= local_query_start:
            events.append(
                _occurrence(
                    row,
                    start=occurrence_start.isoformat(),
                    end=occurrence_end.isoformat() if duration is not None else None,
                )
            )
    return events


def _expand_date_recurring(
    row: dict[str, Any],
    anchor_date: date,
    query_start: datetime,
    query_end: datetime,
    freq: int,
) -> list[CalendarOccurrence]:
    anchor_dt = datetime.combine(anchor_date, datetime_time.min, timezone.utc)
    until_date = _date(row.get("recurrence_until_date"))
    last_query_date = _last_included_date(query_end)
    before_date = min(last_query_date, until_date) if until_date is not None else last_query_date
    rule = rrule(freq, dtstart=anchor_dt, **_rrule_options(row, anchor_dt, row.get("recurrence")))
    events = []
    for occurrence_dt in rule.between(
        anchor_dt,
        datetime.combine(before_date, datetime_time.max, timezone.utc),
        inc=True,
    ):
        occurrence_date = occurrence_dt.date()
        if occurrence_date < anchor_date:
            continue
        if _date_range_overlaps(occurrence_date, occurrence_date, query_start, query_end):
            events.append(_occurrence(row, start_date=occurrence_date.isoformat(), end_date=occurrence_date.isoformat()))
    return events


def _rrule_options(row: dict[str, Any], anchor: datetime, recurrence: str | None) -> dict[str, int]:
    if recurrence == "WEEKLY":
        return {"byweekday": anchor.weekday()}
    if recurrence == "MONTHLY":
        return {"bymonthday": anchor.day}
    if recurrence == "YEARLY":
        return {"bymonth": anchor.month, "bymonthday": anchor.day}
    return {}


def _rrule_frequency(recurrence: str | None) -> int | None:
    return {
        "DAILY": DAILY,
        "WEEKLY": WEEKLY,
        "MONTHLY": MONTHLY,
        "YEARLY": YEARLY,
    }.get(recurrence)


def _recurrence_until_datetime(row: dict[str, Any], tzinfo) -> datetime | None:
    until_at = _aware(row.get("recurrence_until_at"), datetime.now(tzinfo))
    if until_at is not None:
        return until_at.astimezone(tzinfo)
    until_date = _date(row.get("recurrence_until_date"))
    if until_date is None:
        return None
    return datetime.combine(until_date, datetime_time.max, tzinfo)


def _duration(start_at: datetime, end_at: datetime | None) -> timedelta | None:
    if end_at is None:
        return None
    duration = end_at - start_at
    if duration.total_seconds() <= 0:
        return None
    return duration


def _occurrence(
    row: dict[str, Any],
    *,
    start: str | None = None,
    end: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> CalendarOccurrence:
    return CalendarOccurrence(
        calendar_event_id=str(row.get("calendar_event_id")),
        semantic_memory_id=str(row.get("semantic_memory_id")),
        memory_created_at=row.get("memory_created_at").isoformat()
        if hasattr(row.get("memory_created_at"), "isoformat")
        else row.get("memory_created_at"),
        title=row.get("title"),
        event_kind=row.get("event_kind"),
        start=start,
        end=end,
        start_date=start_date,
        end_date=end_date,
        all_day=bool(row.get("all_day")),
        location_text=row.get("location_text"),
        recurring=_is_recurring(row),
        recurrence=row.get("recurrence"),
        recurrence_specifics=row.get("recurrence_specifics"),
        entities=list(row.get("entities") or []),
        evidence=list(row.get("evidence") or []),
    )


def _is_recurring(row: dict[str, Any]) -> bool:
    return bool(row.get("recurrence") and row.get("recurrence") != "NONE")


def _date_range_overlaps(start_date: date, end_date: date, query_start: datetime, query_end: datetime) -> bool:
    return start_date <= _last_included_date(query_end) and end_date >= query_start.date()


def _last_included_date(end: datetime) -> date:
    return (end - timedelta(microseconds=1)).date()


def _aware(value: Any, fallback: datetime) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=fallback.tzinfo)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=fallback.tzinfo)
    return None


def _event_timezone(row: dict[str, Any], fallback) -> Any:
    timezone_text = row.get("timezone_text")
    if timezone_text:
        try:
            return ZoneInfo(timezone_text)
        except ZoneInfoNotFoundError:
            return fallback
    return fallback


def _date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return None


def _sort_key(event: CalendarOccurrence):
    return event.start or event.start_date or ""


def _validate_interval(start: datetime, end: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("CalendarTool requires timezone-aware start and end")
    if start >= end:
        raise ValueError("CalendarTool requires start < end")


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
