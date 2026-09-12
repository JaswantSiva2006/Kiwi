from __future__ import annotations

import json

from kivi_memory.calendar.models import CalendarOccurrence, ScheduleDiagnostics, ScheduleResult
from kivi_memory.read_orchestrator import ReadSourceDecision, ToolExecutionResult, build_context
from kivi_memory.working_memory.models import ThreadMessage


def test_neither_context_contains_thread_and_query_only() -> None:
    context = build_context(
        "Explain your last answer.",
        [message("assistant", "Recursion is when a function calls itself.")],
        orchestrator_result=decision(False, False),
    )

    assert "<CURRENT_THREAD>" in context
    assert "assistant: Recursion is when a function calls itself." in context
    assert "<USER_QUERY>\nExplain your last answer.\n</USER_QUERY>" in context
    assert "<SEMANTIC_MEMORY>" not in context
    assert "<CALENDAR>" not in context


def test_semantic_only_context_includes_structured_memories() -> None:
    context = build_context(
        "Who handles Atlas?",
        [message("user", "Tell me about Atlas.")],
        semantic_memories=[semantic_memory()],
        orchestrator_result=decision(True, False),
    )

    assert "<SEMANTIC_MEMORY>" in context
    assert '"memory_id": "mem-1"' in context
    assert '"canonical_text": "Priya handles Atlas."' in context
    assert "<CALENDAR>" not in context


def test_calendar_only_context_includes_calendar_events() -> None:
    context = build_context(
        "What do I have next Tuesday?",
        [message("user", "Show my schedule.")],
        calendar_result=calendar_result(),
        orchestrator_result=decision(False, True),
    )

    assert "<CALENDAR>" in context
    assert '"title": "Atlas review"' in context
    assert '"start": "2026-09-15T15:00:00+05:30"' in context
    assert '"source_text": "I have an Atlas review every Tuesday at 3 PM."' in context
    assert "<SEMANTIC_MEMORY>" not in context


def test_both_context_includes_semantic_memory_and_calendar() -> None:
    context = build_context(
        "What do I have with Priya next Tuesday?",
        [message("user", "Priya is important here.")],
        semantic_memories=[semantic_memory()],
        calendar_result=calendar_result(),
        orchestrator_result=decision(True, True),
    )

    assert "<CURRENT_THREAD>" in context
    assert "<SEMANTIC_MEMORY>" in context
    assert "<CALENDAR>" in context
    assert context.count("<USER_QUERY>") == 1


def test_calendar_context_filters_dedupes_and_bounds_by_query_relevance() -> None:
    events = [
        calendar_event(
            "old-payments",
            "old-payments-mem",
            "Payments review",
            "2026-04-01T14:00:00+05:30",
            "2026-09-12T08:00:00+00:00",
            "The Payments review is usually Wednesday at 2 PM.",
            recurring=True,
        ),
        calendar_event(
            "old-payments",
            "old-payments-mem",
            "Payments review",
            "2026-04-08T14:00:00+05:30",
            "2026-09-12T08:00:00+00:00",
            "The Payments review is usually Wednesday at 2 PM.",
            recurring=True,
        ),
        calendar_event(
            "exception",
            "exception-mem",
            "Payments review",
            "2026-04-09T11:00:00+05:30",
            "2026-09-12T09:00:00+00:00",
            "This week's Payments review is Thursday at 11 AM instead.",
        ),
        *[
            calendar_event(
                f"other-{index}",
                f"other-mem-{index}",
                f"Unrelated meeting {index}",
                f"2026-05-{index:02d}T09:00:00+05:30",
                f"2026-09-11T{index % 24:02d}:00:00+00:00",
                f"Unrelated meeting {index}.",
            )
            for index in range(1, 35)
        ],
    ]

    context = build_context(
        "What is my normal Payments review schedule, and was there any exception?",
        [],
        calendar_result=schedule_result(events),
        orchestrator_result=decision(False, True),
    )
    payload = calendar_payload(context)
    selected = payload["events"]

    assert payload["total_events"] == 37
    assert payload["returned_events"] == 12
    assert [event["semantic_memory_id"] for event in selected].count("old-payments-mem") == 1
    assert selected[0]["semantic_memory_id"] == "exception-mem"
    assert selected[1]["semantic_memory_id"] == "old-payments-mem"
    assert selected[0]["evidence"][0]["source_text"] == "This week's Payments review is Thursday at 11 AM instead."
    assert "other-mem-1" not in {event["semantic_memory_id"] for event in selected}


def test_context_includes_redis_thread_history_tool_result() -> None:
    context = build_context(
        "What optimization method did I mention earlier?",
        [message("user", "Recent turn.")],
        tool_results=[
            ToolExecutionResult(
                tool="redis_thread_history.search",
                arguments={"query": "optimization"},
                result={
                    "episodes": [
                        {
                            "episode_id": "old-1",
                            "messages": [
                                {"role": "USER", "text": "I mentioned simulated annealing earlier."}
                            ],
                        }
                    ]
                },
                latency_ms=0.2,
            )
        ],
    )

    assert "<REDIS_THREAD_HISTORY>" in context
    assert "simulated annealing" in context
    assert "<SEMANTIC_MEMORY>" not in context
    assert context.count("<USER_QUERY>") == 1


def test_context_includes_web_context_tool_result() -> None:
    context = build_context(
        "What's the latest NVIDIA news?",
        [message("user", "Hi.")],
        tool_results=[
            ToolExecutionResult(
                tool="web.search",
                arguments={"query": "latest NVIDIA news", "freshness": "day"},
                result={
                    "query": "latest NVIDIA news",
                    "sources": [
                        {
                            "source_id": "W1",
                            "title": "NVIDIA announces update",
                            "url": "https://example.com/nvidia",
                            "published_at": "2026-09-11",
                            "relevant_text": "NVIDIA announced a product update today.",
                        }
                    ],
                },
                latency_ms=10.0,
            )
        ],
    )

    assert "<WEB_CONTEXT>" in context
    assert "[W1]" in context
    assert "URL: https://example.com/nvidia" in context
    assert "<USER_QUERY>" in context


def decision(use_semantic: bool, use_calendar: bool) -> ReadSourceDecision:
    return ReadSourceDecision(
        use_semantic_memory=use_semantic,
        use_calendar=use_calendar,
        calendar_start="2026-09-15T00:00:00+05:30" if use_calendar else None,
        calendar_end="2026-09-16T00:00:00+05:30" if use_calendar else None,
    )


def message(role: str, text: str) -> ThreadMessage:
    return ThreadMessage(message_id=f"{role}-1", role=role, text=text, timestamp="2026-09-11T10:00:00+05:30")


def semantic_memory() -> dict:
    return {
        "memory_id": "mem-1",
        "canonical_text": "Priya handles Atlas.",
        "created_at": "2026-09-12T10:00:00+05:30",
        "subject": {"text": "Priya"},
        "predicate_type": "HANDLES",
        "arguments": [{"role": "project", "text": "Atlas"}],
        "temporal": {"temporal_kind": "STATE"},
        "evidence": [{"message_id": "msg-1", "source_text": "Priya handles Atlas."}],
    }


def calendar_result() -> ScheduleResult:
    return schedule_result([calendar_event(
        "cal-1",
        "mem-2",
        "Atlas review",
        "2026-09-15T15:00:00+05:30",
        "2026-09-12T10:00:00+05:30",
        "I have an Atlas review every Tuesday at 3 PM.",
        end="2026-09-15T16:00:00+05:30",
        recurring=False,
        entities=[{"role": "participant", "mention_text": "Priya"}],
    )])


def schedule_result(events: list[CalendarOccurrence]) -> ScheduleResult:
    return ScheduleResult(
        events=events,
        diagnostics=ScheduleDiagnostics(
            db_candidate_query_ms=1.0,
            recurrence_expansion_ms=0.0,
            total_ms=1.0,
            db_recurring_series_considered=0,
            occurrences_returned=1,
        ),
    )


def calendar_event(
    calendar_event_id: str,
    semantic_memory_id: str,
    title: str,
    start: str,
    memory_created_at: str,
    source_text: str,
    *,
    end: str | None = None,
    recurring: bool = False,
    entities: list[dict] | None = None,
) -> CalendarOccurrence:
    return CalendarOccurrence(
        calendar_event_id=calendar_event_id,
        semantic_memory_id=semantic_memory_id,
        memory_created_at=memory_created_at,
        title=title,
        event_kind="MEETING",
        start=start,
        end=end,
        recurring=recurring,
        recurrence="WEEKLY" if recurring else "NONE",
        recurrence_specifics="WEDNESDAY 2 PM" if recurring else None,
        entities=entities or [],
        evidence=[
            {
                "episode_id": "corpus-rec-0010",
                "message_id": "m1",
                "source_text": source_text,
                "observed_at": "2026-01-07T11:54:00+05:30",
            }
        ],
    )


def calendar_payload(context: str) -> dict:
    start = context.index("<CALENDAR>") + len("<CALENDAR>")
    end = context.index("</CALENDAR>")
    return json.loads(context[start:end].strip())
