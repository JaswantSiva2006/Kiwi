from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import pytest

from kivi_memory.common.schemas import EpisodeMessage, MemoryEpisode
from kivi_memory.ledger.models import AddMemoryResult, MemoryNotFoundError
from kivi_memory.ledger.mutations import execute_reconciliation


@dataclass
class FakeCursor:
    db: "FakeMutationRepository"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


class FakeConnection:
    def __init__(self, repo: "FakeMutationRepository") -> None:
        self.repo = repo
        self.snapshot = None

    def __enter__(self):
        self.snapshot = self.repo.snapshot()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.repo.restore(self.snapshot)
        return None

    def cursor(self):
        return FakeCursor(self.repo)


class FakeMutationRepository:
    def __init__(self) -> None:
        self.memories = {}
        self.evidence = {}
        self.events = {}
        self.next_id = 1
        self.fail_event = False

    def connect(self):
        return FakeConnection(self)

    def snapshot(self):
        return deepcopy((self.memories, self.evidence, self.events, self.next_id))

    def restore(self, snapshot) -> None:
        self.memories, self.evidence, self.events, self.next_id = deepcopy(snapshot)

    def insert_memory(self, values) -> AddMemoryResult:
        memory_id = f"memory-{self.next_id}"
        self.next_id += 1
        self.memories[memory_id] = {
            "memory_id": memory_id,
            "canonical_text": values["memory_values"]["canonical_text"],
            "status": "ACTIVE",
            "version": 1,
        }
        self.evidence[memory_id] = list(values["evidence_values"])
        self.events[memory_id] = [{"event_type": "MEMORY_ADDED", "payload": values["event_payload"]}]
        return AddMemoryResult(memory_id=memory_id, status="ACTIVE", version=1)

    def lock_active_memory(self, cur, memory_id: str):
        del cur
        row = self.memories.get(memory_id)
        if row is None:
            raise MemoryNotFoundError(f"memory does not exist: {memory_id}")
        if row["status"] != "ACTIVE":
            raise ValueError(f"target memory is not ACTIVE: {memory_id}")
        return row

    def append_evidence_in_transaction(self, cur, *, memory_id, evidence_values):
        del cur
        existing = self.evidence.setdefault(memory_id, [])
        inserted = 0
        for evidence in evidence_values:
            key = (evidence["episode_id"], evidence["message_id"], evidence["source_text"])
            if key not in {(row["episode_id"], row["message_id"], row["source_text"]) for row in existing}:
                existing.append(evidence)
                inserted += 1
        return inserted

    def increment_memory_version_in_transaction(self, cur, *, memory_id):
        del cur
        self.memories[memory_id]["version"] += 1

    def update_memory_status_in_transaction(self, cur, *, memory_id, status):
        del cur
        self.memories[memory_id]["status"] = status
        self.memories[memory_id]["version"] += 1

    def insert_event_in_transaction(self, cur, *, memory_id, event_type, payload):
        del cur
        if self.fail_event:
            raise RuntimeError("event insert failed")
        self.events.setdefault(memory_id, []).append({"event_type": event_type, "payload": payload})


def fake_insert(cur, **values):
    return cur.db.insert_memory(values)


def test_add_creates_active_memory_and_indexes(monkeypatch) -> None:
    repo = FakeMutationRepository()
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    indexed = []
    graphed = []

    result = execute_reconciliation(
        {"op": "ADD", "targets": []},
        incoming("User plays badminton."),
        [],
        None,
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: indexed.append(memory_id) or {"indexed": True},
        graph_projector=lambda memory_id: graphed.append(memory_id) or {"projected": True},
    )

    assert result.created_memory_id == "memory-1"
    assert repo.memories["memory-1"]["status"] == "ACTIVE"
    assert indexed == ["memory-1"]
    assert graphed == ["memory-1"]


def test_add_calendar_event_creates_calendar_projection(monkeypatch) -> None:
    repo = FakeMutationRepository()
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    calendar_calls = []
    monkeypatch.setattr(
        "kivi_memory.ledger.mutations.create_calendar_event_for_memory",
        lambda cur, *, memory_id, assertion, temporal_metadata, entity_resolution, **kwargs: (
            calendar_calls.append(
                {
                    "memory_id": memory_id,
                    "memory_type": assertion["memory_type"],
                    "temporal": temporal_metadata,
                    "entity_resolution": entity_resolution,
                }
            )
            or "calendar-1"
        ),
    )

    result = execute_reconciliation(
        {"op": "ADD", "targets": []},
        calendar_incoming(),
        [],
        temporal_event("2026-09-07T15:00:00+05:30"),
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: {"indexed": memory_id},
        graph_projector=lambda memory_id: {"projected": memory_id},
    )

    assert result.created_memory_id == "memory-1"
    assert result.calendar_event_id == "calendar-1"
    assert calendar_calls == [
        {
            "memory_id": "memory-1",
            "memory_type": "CALENDAR_EVENT",
            "temporal": temporal_event("2026-09-07T15:00:00+05:30"),
            "entity_resolution": entity_resolution(),
        }
    ]


def test_reinforce_adds_evidence_and_event_without_new_memory(monkeypatch) -> None:
    repo = seeded_repo("old-1", "Priya handles Atlas.")
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    monkeypatch.setattr(
        "kivi_memory.ledger.mutations.create_calendar_event_for_memory",
        lambda *args, **kwargs: pytest.fail("REINFORCE must not create calendar projections"),
    )

    result = execute_reconciliation(
        {"op": "REINFORCE", "targets": ["C1"]},
        incoming("Priya is responsible for Atlas."),
        [{"memory_id": "old-1"}],
        None,
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: pytest.fail("REINFORCE must not re-index"),
        graph_projector=lambda memory_id: pytest.fail("REINFORCE must not project graph links"),
    )

    assert result.created_memory_id is None
    assert set(repo.memories) == {"old-1"}
    assert repo.memories["old-1"]["version"] == 2
    assert repo.events["old-1"][-1]["event_type"] == "MEMORY_REINFORCED"


def test_supersede_calendar_marks_old_and_creates_new_projection(monkeypatch) -> None:
    repo = seeded_repo("old-1", "The Atlas review is tomorrow.")
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    status_updates = []
    created = []
    monkeypatch.setattr(
        "kivi_memory.ledger.mutations.mark_calendar_event_superseded",
        lambda cur, *, memory_id: status_updates.append(("SUPERSEDED", memory_id)) or 1,
    )
    monkeypatch.setattr(
        "kivi_memory.ledger.mutations.create_calendar_event_for_memory",
        lambda cur, *, memory_id, assertion, temporal_metadata, entity_resolution, **kwargs: created.append(memory_id) or "calendar-2",
    )

    result = execute_reconciliation(
        {"op": "SUPERSEDE", "targets": ["C1"]},
        calendar_incoming("The Atlas review moved to tomorrow at 4 PM."),
        [{"memory_id": "old-1"}],
        temporal_event("2026-09-07T16:00:00+05:30"),
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: {"indexed": memory_id},
        graph_projector=lambda memory_id: {"projected": memory_id},
    )

    assert repo.memories["old-1"]["status"] == "SUPERSEDED"
    assert status_updates == [("SUPERSEDED", "old-1")]
    assert created == [result.created_memory_id]
    assert result.calendar_event_id == "calendar-2"
    assert result.calendar_status_updates == [{"memory_id": "old-1", "status": "SUPERSEDED", "updated": 1}]


def test_supersede_marks_old_and_links_replacement(monkeypatch) -> None:
    repo = seeded_repo("old-1", "User likes black coffee.")
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    graphed = []

    result = execute_reconciliation(
        {"op": "SUPERSEDE", "targets": ["C1"]},
        incoming("User no longer likes black coffee."),
        [{"memory_id": "old-1"}],
        None,
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: {"indexed": memory_id},
        graph_projector=lambda memory_id: graphed.append(memory_id) or {"projected": memory_id},
    )

    assert repo.memories["old-1"]["status"] == "SUPERSEDED"
    assert repo.memories[result.created_memory_id]["status"] == "ACTIVE"
    assert repo.events["old-1"][-1]["payload"]["replacement_memory_id"] == result.created_memory_id
    assert graphed == [result.created_memory_id]


def test_retract_calendar_cancels_old_without_scheduled_cancellation(monkeypatch) -> None:
    repo = seeded_repo("old-1", "The Atlas review is tomorrow.")
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    cancelled = []
    monkeypatch.setattr(
        "kivi_memory.ledger.mutations.cancel_calendar_event_for_memory",
        lambda cur, *, memory_id: cancelled.append(memory_id) or 1,
    )
    monkeypatch.setattr(
        "kivi_memory.ledger.mutations.create_calendar_event_for_memory",
        lambda *args, **kwargs: pytest.fail("RETRACT must not create a scheduled cancellation"),
    )

    result = execute_reconciliation(
        {"op": "RETRACT", "targets": ["C1"]},
        calendar_incoming("Cancel the Atlas review tomorrow."),
        [{"memory_id": "old-1"}],
        temporal_event("2026-09-07T15:00:00+05:30"),
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: {"indexed": memory_id},
        graph_projector=lambda memory_id: {"projected": memory_id},
    )

    assert repo.memories["old-1"]["status"] == "RETRACTED"
    assert cancelled == ["old-1"]
    assert result.calendar_event_id is None
    assert result.calendar_status_updates == [{"memory_id": "old-1", "status": "CANCELLED", "updated": 1}]


def test_retract_marks_old_and_links_correction(monkeypatch) -> None:
    repo = seeded_repo("old-1", "User likes black coffee.")
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    graphed = []

    result = execute_reconciliation(
        {"op": "RETRACT", "targets": ["C1"]},
        incoming("That earlier statement was wrong; user never liked black coffee."),
        [{"memory_id": "old-1"}],
        None,
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: {"indexed": memory_id},
        graph_projector=lambda memory_id: graphed.append(memory_id) or {"projected": memory_id},
    )

    assert repo.memories["old-1"]["status"] == "RETRACTED"
    assert repo.events["old-1"][-1]["payload"]["correction_memory_id"] == result.created_memory_id
    assert graphed == [result.created_memory_id]


def test_no_memory_performs_zero_ledger_changes() -> None:
    repo = FakeMutationRepository()

    result = execute_reconciliation(
        {"op": "NO_MEMORY", "targets": []},
        incoming("scratch"),
        [],
        None,
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        graph_projector=lambda memory_id: pytest.fail("NO_MEMORY must not project graph links"),
    )

    assert result.executed is False
    assert repo.memories == {}


def test_target_not_found_fails_cleanly() -> None:
    with pytest.raises(MemoryNotFoundError):
        execute_reconciliation(
            {"op": "REINFORCE", "targets": ["C1"]},
            incoming("fact"),
            [{"memory_id": "missing"}],
            None,
            entity_resolution(),
            episode(),
            validation_report=valid_report(),
            repository=FakeMutationRepository(),
        )


def test_target_already_non_active_fails() -> None:
    repo = seeded_repo("old-1", "fact", status="SUPERSEDED")

    with pytest.raises(ValueError, match="not ACTIVE"):
        execute_reconciliation(
            {"op": "REINFORCE", "targets": ["C1"]},
            incoming("fact"),
            [{"memory_id": "old-1"}],
            None,
            entity_resolution(),
            episode(),
            validation_report=valid_report(),
            repository=repo,
        )


def test_invalid_c_number_fails() -> None:
    with pytest.raises(ValueError, match="invalid reconciliation target"):
        execute_reconciliation(
            {"op": "REINFORCE", "targets": ["C9"]},
            incoming("fact"),
            [{"memory_id": "old-1"}],
            None,
            entity_resolution(),
            episode(),
            validation_report=valid_report(),
            repository=seeded_repo("old-1", "fact"),
        )


def test_rollback_if_new_memory_insert_fails(monkeypatch) -> None:
    repo = seeded_repo("old-1", "fact")

    def fail_insert(cur, **values):
        del cur, values
        raise RuntimeError("insert failed")

    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fail_insert)

    with pytest.raises(RuntimeError, match="insert failed"):
        execute_reconciliation(
            {"op": "SUPERSEDE", "targets": ["C1"]},
            incoming("new fact"),
            [{"memory_id": "old-1"}],
            None,
            entity_resolution(),
            episode(),
            validation_report=valid_report(),
            repository=repo,
        )

    assert repo.memories == {"old-1": {"memory_id": "old-1", "canonical_text": "fact", "status": "ACTIVE", "version": 1}}


def test_rollback_if_event_insert_fails(monkeypatch) -> None:
    repo = seeded_repo("old-1", "fact")
    repo.fail_event = True
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)

    with pytest.raises(RuntimeError, match="event insert failed"):
        execute_reconciliation(
            {"op": "RETRACT", "targets": ["C1"]},
            incoming("correction"),
            [{"memory_id": "old-1"}],
            None,
            entity_resolution(),
            episode(),
            validation_report=valid_report(),
            repository=repo,
        )

    assert set(repo.memories) == {"old-1"}
    assert repo.memories["old-1"]["status"] == "ACTIVE"


def test_failed_embedding_projection_does_not_undo_ledger_commit(monkeypatch) -> None:
    repo = FakeMutationRepository()
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)

    result = execute_reconciliation(
        {"op": "ADD", "targets": []},
        incoming("fact"),
        [],
        None,
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: (_ for _ in ()).throw(RuntimeError("embedding failed")),
        graph_projector=lambda memory_id: {"projected": memory_id},
    )

    assert result.created_memory_id == "memory-1"
    assert result.embedding_projection_error == "embedding failed"
    assert repo.memories["memory-1"]["status"] == "ACTIVE"


def test_failed_graph_projection_does_not_undo_ledger_commit(monkeypatch) -> None:
    repo = FakeMutationRepository()
    monkeypatch.setattr("kivi_memory.ledger.mutations.insert_memory_record_in_transaction", fake_insert)
    indexed = []

    result = execute_reconciliation(
        {"op": "ADD", "targets": []},
        incoming("fact"),
        [],
        None,
        entity_resolution(),
        episode(),
        validation_report=valid_report(),
        repository=repo,
        indexer=lambda memory_id: indexed.append(memory_id) or {"indexed": memory_id},
        graph_projector=lambda memory_id: (_ for _ in ()).throw(RuntimeError("graph failed")),
    )

    assert result.created_memory_id == "memory-1"
    assert result.graph_projection_error == "graph failed"
    assert indexed == ["memory-1"]
    assert repo.memories["memory-1"]["status"] == "ACTIVE"


def seeded_repo(memory_id: str, text: str, status: str = "ACTIVE") -> FakeMutationRepository:
    repo = FakeMutationRepository()
    repo.memories[memory_id] = {"memory_id": memory_id, "canonical_text": text, "status": status, "version": 1}
    repo.evidence[memory_id] = []
    repo.events[memory_id] = [{"event_type": "MEMORY_ADDED", "payload": {}}]
    return repo


def incoming(text: str) -> dict:
    return {
        "canonical_text": text,
        "memory_type": "PREFERENCE",
        "subject": {"text": "user", "entity_type": "person"},
        "semantic_arguments": [],
        "predicate_type": "LIKES",
        "modality": "FACT",
        "polarity": "POSITIVE",
        "certainty": "CERTAIN",
        "explicitness": "EXPLICIT",
        "attributed_to": "user",
        "source_spans": [{"message_id": "m1", "text": text}],
    }


def calendar_incoming(text: str = "The Atlas design review is tomorrow at 3 PM.") -> dict:
    return {
        "canonical_text": text,
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
        "source_spans": [{"message_id": "m1", "text": text}],
        "calendar_event": {
            "title": "Atlas design review",
            "event_kind": "MEETING",
            "location_text": None,
            "start_time_text": "tomorrow at 3 PM",
            "end_time_text": None,
            "duration_text": None,
            "recurrence_text": None,
            "timezone_text": None,
            "all_day_hint": False,
        },
    }


def temporal_event(event_time: str) -> dict:
    return {
        "temporal_kind": "DISCRETE_EVENT",
        "valid_from_hint": None,
        "valid_to_hint": None,
        "event_time": event_time,
        "temporal_precision": "EXACT",
        "recurrence": "NONE",
        "recurrence_specifics": None,
    }


def entity_resolution() -> dict:
    return {"subject": {"resolution": "MATCHED", "entity_id": None}, "semantic_arguments": []}


def episode() -> MemoryEpisode:
    return MemoryEpisode(
        episode_id="ep-1",
        messages=[
            EpisodeMessage(
                message_id="m1",
                role="USER",
                timestamp=datetime.fromisoformat("2026-09-10T09:00:00+05:30"),
                text="test",
            )
        ],
    )


def valid_report() -> dict:
    return {"valid": True, "issues": []}
