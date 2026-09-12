from __future__ import annotations

from kivi_memory.calendar.projector import (
    cancel_calendar_event_for_memory,
    create_calendar_event_for_memory,
    mark_calendar_event_superseded,
)
from kivi_memory.common.schemas import CandidateSemanticAssertion, MemoryEpisode, TemporalMetadata


class FakeCalendarRepository:
    def __init__(self) -> None:
        self.events = {}
        self.links = []
        self.status_updates = []

    def insert_calendar_event_in_transaction(self, cur, *, event_values, entity_links):
        del cur
        calendar_event_id = f"calendar-{len(self.events) + 1}"
        self.events[calendar_event_id] = dict(event_values)
        self.links.extend(entity_links)
        return calendar_event_id

    def update_calendar_status_for_memory_in_transaction(self, cur, *, memory_id, status):
        del cur
        self.status_updates.append({"memory_id": memory_id, "status": status})
        return 1


def test_create_calendar_event_uses_event_time_and_entity_links() -> None:
    repo = FakeCalendarRepository()

    event_id = create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion=calendar_assertion(),
        temporal_metadata=TemporalMetadata(
            temporal_kind="DISCRETE_EVENT",
            event_time="2026-09-07T15:00:00+05:30",
            temporal_precision="EXACT",
            recurrence="NONE",
        ),
        entity_resolution=entity_resolution(),
        repository=repo,
    )

    assert event_id == "calendar-1"
    row = repo.events[event_id]
    assert row["semantic_memory_id"] == "memory-1"
    assert row["title"] == "Atlas design review"
    assert row["start_at"] == "2026-09-07T15:00:00+05:30"
    assert row["start_date"] is None
    assert row["status"] == "SCHEDULED"
    assert {"entity_id": "user-1", "role": "SUBJECT", "mention_text": "user"} in repo.links
    assert {"entity_id": "atlas-1", "role": "PROJECT", "mention_text": "Atlas"} in repo.links


def test_calendar_date_only_does_not_manufacture_midnight_time() -> None:
    repo = FakeCalendarRepository()

    event_id = create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion=calendar_assertion(all_day_hint=True),
        temporal_metadata=TemporalMetadata(
            temporal_kind="DISCRETE_EVENT",
            event_time="2026-09-14",
            temporal_precision="DAY",
            recurrence="NONE",
        ),
        entity_resolution=entity_resolution(),
        repository=repo,
    )

    row = repo.events[event_id]
    assert row["start_at"] is None
    assert row["start_date"] == "2026-09-14"
    assert row["all_day"] is True


def test_recurring_calendar_event_anchors_to_described_weekday_not_observation_time() -> None:
    repo = FakeCalendarRepository()

    event_id = create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion=recurring_calendar_assertion(
            "The user checks in with Vikram about Beacon every Monday morning.",
            "I usually check in with Vikram about Beacon every Monday morning.",
        ),
        temporal_metadata=TemporalMetadata(
            temporal_kind="RECURRENCE",
            temporal_precision="DAY",
            recurrence="WEEKLY",
            recurrence_specifics="MONDAY MORNING",
        ),
        entity_resolution={"subject": {"resolution": "MATCHED", "entity_id": "user-1"}, "semantic_arguments": []},
        episode=episode("I usually check in with Vikram about Beacon every Monday morning.", "2026-02-11T03:15:00+05:30"),
        repository=repo,
    )

    row = repo.events[event_id]
    assert row["start_at"] == "2026-02-16T09:00:00+05:30"
    assert row["start_date"] is None
    assert row["recurrence"] == "WEEKLY"
    assert row["recurrence_specifics"] == "MONDAY MORNING"


def test_recurring_calendar_event_anchors_to_explicit_clock_time() -> None:
    repo = FakeCalendarRepository()

    event_id = create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion=recurring_calendar_assertion(
            "The user has an Atlas review every Tuesday at 3 PM.",
            "I have an Atlas review every Tuesday at 3 PM.",
        ),
        temporal_metadata=TemporalMetadata(
            temporal_kind="RECURRENCE",
            temporal_precision="EXACT",
            recurrence="WEEKLY",
            recurrence_specifics="TUESDAY 3 PM",
        ),
        entity_resolution={"subject": {"resolution": "MATCHED", "entity_id": "user-1"}, "semantic_arguments": []},
        episode=episode("I have an Atlas review every Tuesday at 3 PM.", "2026-01-07T11:54:00+05:30"),
        repository=repo,
    )

    row = repo.events[event_id]
    assert row["start_at"] == "2026-01-13T15:00:00+05:30"
    assert row["start_date"] is None


def test_recurring_calendar_event_prefers_described_weekday_over_bad_temporal_anchor() -> None:
    repo = FakeCalendarRepository()

    event_id = create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion=recurring_calendar_assertion(
            "The Payments review is usually Wednesday at 2 PM.",
            "The Payments review is usually Wednesday at 2 PM.",
        ),
        temporal_metadata=TemporalMetadata(
            temporal_kind="RECURRENCE",
            valid_from_hint="2026-03-31T14:00:00+05:30",
            temporal_precision="EXACT",
            recurrence="WEEKLY",
            recurrence_specifics="WEDNESDAY 2 PM",
        ),
        entity_resolution={"subject": {"resolution": "MATCHED", "entity_id": "payments-1"}, "semantic_arguments": []},
        episode=episode("The Payments review is usually Wednesday at 2 PM.", "2026-03-30T16:30:00+05:30"),
        repository=repo,
    )

    row = repo.events[event_id]
    assert row["start_at"] == "2026-04-01T14:00:00+05:30"


def test_non_calendar_and_negative_calendar_assertions_create_no_scheduled_event() -> None:
    repo = FakeCalendarRepository()
    assert create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion={**calendar_assertion().model_dump(), "memory_type": "PREFERENCE"},
        temporal_metadata=None,
        entity_resolution=entity_resolution(),
        repository=repo,
    ) is None
    assert create_calendar_event_for_memory(
        object(),
        memory_id="memory-2",
        assertion={**calendar_assertion().model_dump(mode="json"), "polarity": "NEGATIVE"},
        temporal_metadata=None,
        entity_resolution=entity_resolution(),
        repository=repo,
    ) is None
    assert repo.events == {}


def test_calendar_event_without_payload_derives_minimal_projection_row() -> None:
    repo = FakeCalendarRepository()

    event_id = create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion=bare_calendar_assertion(
            "The Payments review is scheduled for Thursday at 11 AM this week.",
            [{"role": "time", "text": "Thursday at 11 AM this week", "is_entity": False, "entity_type": None}],
        ),
        temporal_metadata=TemporalMetadata(
            temporal_kind="DISCRETE_EVENT",
            event_time=None,
            temporal_precision="EXACT",
            recurrence="NONE",
        ),
        entity_resolution={"subject": {"resolution": "MATCHED", "entity_id": "payments-1"}, "semantic_arguments": []},
        episode=episode("This week's Payments review is Thursday at 11 AM instead.", "2026-04-03T21:15:00+05:30"),
        repository=repo,
    )

    row = repo.events[event_id]
    assert row["title"] == "Payments review"
    assert row["start_at"] == "2026-04-09T11:00:00+05:30"
    assert row["recurrence"] == "NONE"


def test_calendar_event_with_unparseable_time_still_projects_row_for_searchability() -> None:
    repo = FakeCalendarRepository()

    event_id = create_calendar_event_for_memory(
        object(),
        memory_id="memory-1",
        assertion=bare_calendar_assertion(
            "The planning review is scheduled after the launch checkpoint.",
            [{"role": "time", "text": "after the launch checkpoint", "is_entity": False, "entity_type": None}],
        ),
        temporal_metadata=TemporalMetadata(
            temporal_kind="DISCRETE_EVENT",
            event_time=None,
            temporal_precision="APPROXIMATE",
            recurrence="NONE",
        ),
        entity_resolution={"subject": {"resolution": "MATCHED", "entity_id": "review-1"}, "semantic_arguments": []},
        repository=repo,
    )

    row = repo.events[event_id]
    assert row["semantic_memory_id"] == "memory-1"
    assert row["title"] == "Planning review"
    assert row["start_at"] is None
    assert row["start_date"] is None
    assert row["status"] == "SCHEDULED"


def test_supersede_and_cancel_update_existing_calendar_status() -> None:
    repo = FakeCalendarRepository()

    assert mark_calendar_event_superseded(object(), memory_id="old-1", repository=repo) == 1
    assert cancel_calendar_event_for_memory(object(), memory_id="old-2", repository=repo) == 1

    assert repo.status_updates == [
        {"memory_id": "old-1", "status": "SUPERSEDED"},
        {"memory_id": "old-2", "status": "CANCELLED"},
    ]


def calendar_assertion(all_day_hint: bool = False) -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": "The user's Atlas design review is tomorrow at 3 PM.",
            "memory_type": "CALENDAR_EVENT",
            "subject": {"text": "user", "entity_type": "person"},
            "semantic_arguments": [
                {"role": "project", "text": "Atlas", "is_entity": True, "entity_type": "PROJECT"},
                {"role": "time", "text": "tomorrow at 3 PM", "is_entity": False, "entity_type": None},
            ],
            "predicate_type": "HAS_SCHEDULED_EVENT",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "user",
            "source_spans": [{"message_id": "m1", "text": "I have the Atlas design review tomorrow at 3 PM."}],
            "calendar_event": {
                "title": "Atlas design review",
                "event_kind": "MEETING",
                "location_text": None,
                "start_time_text": "tomorrow at 3 PM",
                "end_time_text": None,
                "duration_text": None,
                "recurrence_text": None,
                "timezone_text": None,
                "all_day_hint": all_day_hint,
            },
        }
    )


def recurring_calendar_assertion(text: str, source_text: str) -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": text,
            "memory_type": "CALENDAR_EVENT",
            "subject": {"text": "user", "entity_type": "person"},
            "semantic_arguments": [],
            "predicate_type": "HAS_RECURRING_EVENT",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "user",
            "source_spans": [{"message_id": "m1", "text": source_text}],
            "calendar_event": {
                "title": "Recurring event",
                "event_kind": "MEETING",
                "location_text": None,
                "start_time_text": source_text,
                "end_time_text": None,
                "duration_text": None,
                "recurrence_text": source_text,
                "timezone_text": "Asia/Kolkata",
                "all_day_hint": False,
            },
        }
    )


def bare_calendar_assertion(text: str, arguments: list[dict]) -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": text,
            "memory_type": "CALENDAR_EVENT",
            "subject": {"text": "Payments review", "entity_type": "event"},
            "semantic_arguments": arguments,
            "predicate_type": "SCHEDULED",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "user",
            "source_spans": [{"message_id": "m1", "text": text}],
            "calendar_event": None,
        }
    )


def episode(text: str, timestamp: str) -> MemoryEpisode:
    return MemoryEpisode.model_validate(
        {
            "episode_id": "episode-1",
            "session_id": "session-1",
            "timezone": "Asia/Kolkata",
            "locale": "en-IN",
            "messages": [{"message_id": "m1", "role": "USER", "timestamp": timestamp, "text": text}],
        }
    )


def entity_resolution() -> dict:
    return {
        "subject": {"resolution": "MATCHED", "entity_id": "user-1"},
        "semantic_arguments": [
            {"role": "project", "result": {"resolution": "MATCHED", "entity_id": "atlas-1"}},
        ],
    }
