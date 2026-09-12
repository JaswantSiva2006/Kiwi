from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kivi_memory.ledger import MemoryNotFoundError
from kivi_memory.ledger.repository import LedgerRepository


class FakeLedgerDatabase:
    def __init__(self) -> None:
        self.memories = {}
        self.arguments = []
        self.evidence = []
        self.events = []
        self.fail_on_evidence = False

    def connect(self, database_url):
        return FakeConnection(self)

    def snapshot(self):
        return (
            self.memories.copy(),
            [row.copy() for row in self.arguments],
            [row.copy() for row in self.evidence],
            [row.copy() for row in self.events],
        )

    def restore(self, snapshot) -> None:
        self.memories, self.arguments, self.evidence, self.events = snapshot


class FakeConnection:
    def __init__(self, db: FakeLedgerDatabase) -> None:
        self.db = db
        self.snapshot = None

    def __enter__(self):
        self.snapshot = self.db.snapshot()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.db.restore(self.snapshot)

    def cursor(self):
        return FakeCursor(self.db)


class FakeCursor:
    def __init__(self, db: FakeLedgerDatabase) -> None:
        self.db = db
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, sql: str, params: tuple) -> None:
        normalized_sql = " ".join(sql.split()).lower()
        if normalized_sql.startswith("insert into semantic_memories"):
            self.db.memories["memory-1"] = {"memory_id": "memory-1", "status": "ACTIVE", "version": 1}
            self.result = self.db.memories["memory-1"]
            return
        if normalized_sql.startswith("insert into memory_arguments"):
            self.db.arguments.append({"memory_id": params[0], "role": params[1]})
            self.result = None
            return
        if normalized_sql.startswith("insert into memory_evidence"):
            if self.db.fail_on_evidence:
                raise RuntimeError("evidence insert failed")
            self.db.evidence.append({"memory_id": params[0], "message_id": params[2]})
            self.result = None
            return
        if normalized_sql.startswith("insert into memory_events"):
            self.db.events.append({"memory_id": params[0], "event_type": params[1]})
            self.result = None
            return
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self.result


def test_repository_transaction_rollback_leaves_no_partial_rows(monkeypatch) -> None:
    db = FakeLedgerDatabase()
    db.fail_on_evidence = True
    monkeypatch.setattr("kivi_memory.ledger.repository.connect", db.connect)

    with pytest.raises(RuntimeError, match="evidence insert failed"):
        LedgerRepository().add_memory_record(
            memory_values=memory_values(),
            argument_values=[{"role": "person", "text": "Kavya", "is_entity": True, "entity_id": "e-1", "entity_type": "person", "position": 0}],
            evidence_values=[{"episode_id": "ep-1", "message_id": "m1", "source_text": "I met Kavya yesterday.", "observed_at": "2026-09-10T09:00:00+05:30"}],
            event_payload={"version": 1, "status": "ACTIVE", "source_episode_id": "ep-1"},
        )

    assert db.memories == {}
    assert db.arguments == []
    assert db.evidence == []
    assert db.events == []


def memory_values() -> dict:
    return {
        "canonical_text": "The user met Kavya yesterday.",
        "subject_entity_id": "user-1",
        "subject_text": "the user",
        "subject_entity_type": "person",
        "predicate_type": "MET",
        "memory_type": "IMPORTANT_EVENT",
        "modality": "FACT",
        "polarity": "POSITIVE",
        "certainty": "CERTAIN",
        "explicitness": "EXPLICIT",
        "attributed_to": "user",
        "temporal_kind": "DISCRETE_EVENT",
        "valid_from": None,
        "valid_to": None,
        "event_time": "2026-09-09T09:00:00+05:30",
        "temporal_precision": "DAY",
        "recurrence": "NONE",
        "recurrence_specifics": None,
    }


class FakeReadDatabase:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.memories = [
            summary_row("memory-1", "First", "ACTIVE", "IMPORTANT_EVENT", "entity-1", 3),
            summary_row("memory-2", "Second", "ARCHIVED", "PROJECT_GOAL_TOPIC", "entity-2", 2),
            summary_row("memory-3", "Third", "ACTIVE", "PROJECT_GOAL_TOPIC", "entity-1", 1),
        ]
        self.arguments = {
            "memory-1": [
                {"argument_id": "arg-2", "memory_id": "memory-1", "role": "b", "text": "B", "is_entity": False, "entity_id": None, "entity_type": None, "position": 1, "created_at": dt(4)},
                {"argument_id": "arg-1", "memory_id": "memory-1", "role": "a", "text": "A", "is_entity": True, "entity_id": "entity-a", "entity_type": "person", "position": 0, "created_at": dt(4)},
            ]
        }
        self.evidence = {
            "memory-1": [
                {"evidence_id": "ev-2", "memory_id": "memory-1", "episode_id": "ep-1", "message_id": "m2", "source_text": "second", "observed_at": dt(2), "created_at": dt(6)},
                {"evidence_id": "ev-1", "memory_id": "memory-1", "episode_id": "ep-1", "message_id": "m1", "source_text": "first", "observed_at": dt(1), "created_at": dt(5)},
            ]
        }
        self.events = {
            "memory-1": [
                {"event_id": "event-2", "memory_id": "memory-1", "event_type": "UPDATED", "payload": {}, "created_at": dt(8)},
                {"event_id": "event-1", "memory_id": "memory-1", "event_type": "MEMORY_ADDED", "payload": {}, "created_at": dt(7)},
            ]
        }

    def connect(self, database_url):
        return FakeReadConnection(self)


class FakeReadConnection:
    def __init__(self, db: FakeReadDatabase) -> None:
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def cursor(self):
        return FakeReadCursor(self.db)


class FakeReadCursor:
    def __init__(self, db: FakeReadDatabase) -> None:
        self.db = db
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, sql: str, params: tuple) -> None:
        normalized_sql = " ".join(sql.split()).lower()
        self.db.operations.append(normalized_sql)
        if any(keyword in normalized_sql for keyword in ["insert ", "update ", "delete "]):
            raise AssertionError(f"read helper attempted write SQL: {sql}")

        if normalized_sql.startswith("select * from semantic_memories where memory_id"):
            memory_id = params[0]
            self.result = next((row for row in self.db.memories if row["memory_id"] == memory_id), None)
            return
        if normalized_sql.startswith("select * from memory_arguments"):
            self.result = sorted(self.db.arguments.get(params[0], []), key=lambda row: row["position"])
            return
        if normalized_sql.startswith("select * from memory_evidence"):
            self.result = sorted(self.db.evidence.get(params[0], []), key=lambda row: (row["created_at"], row["evidence_id"]))
            return
        if normalized_sql.startswith("select * from memory_events"):
            self.result = sorted(self.db.events.get(params[0], []), key=lambda row: (row["created_at"], row["event_id"]))
            return
        if normalized_sql.startswith("select memory_id"):
            rows = list(self.db.memories)
            param_index = 0
            if "where" in normalized_sql and "status = %s" in normalized_sql:
                status = params[param_index]
                param_index += 1
                rows = [row for row in rows if row["status"] == status]
            if "where" in normalized_sql and "memory_type = %s" in normalized_sql:
                memory_type = params[param_index]
                param_index += 1
                rows = [row for row in rows if row["memory_type"] == memory_type]
            if "where" in normalized_sql and "subject_entity_id = %s" in normalized_sql:
                subject_entity_id = params[param_index]
                param_index += 1
                rows = [row for row in rows if row["subject_entity_id"] == subject_entity_id]
            limit = params[-1]
            self.result = sorted(rows, key=lambda row: row["created_at"], reverse=True)[:limit]
            return
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result


@pytest.fixture
def fake_read_db(monkeypatch) -> FakeReadDatabase:
    db = FakeReadDatabase()
    monkeypatch.setattr("kivi_memory.ledger.repository.connect", db.connect)
    return db


def test_get_memory_returns_main_row_and_ordered_children(fake_read_db: FakeReadDatabase) -> None:
    stored = LedgerRepository().get_memory("memory-1")

    assert stored.memory["memory_id"] == "memory-1"
    assert [row["position"] for row in stored.arguments] == [0, 1]
    assert [row["evidence_id"] for row in stored.evidence] == ["ev-1", "ev-2"]
    assert [row["event_id"] for row in stored.events] == ["event-1", "event-2"]


def test_get_memory_nonexistent_memory_fails_cleanly(fake_read_db: FakeReadDatabase) -> None:
    with pytest.raises(MemoryNotFoundError, match="memory does not exist"):
        LedgerRepository().get_memory("missing")


def test_list_memories_returns_newest_first(fake_read_db: FakeReadDatabase) -> None:
    memories = LedgerRepository().list_memories()

    assert [row["memory_id"] for row in memories] == ["memory-1", "memory-2", "memory-3"]


def test_list_memories_limit_works(fake_read_db: FakeReadDatabase) -> None:
    memories = LedgerRepository().list_memories(limit=2)

    assert [row["memory_id"] for row in memories] == ["memory-1", "memory-2"]


def test_list_memories_status_filter_works(fake_read_db: FakeReadDatabase) -> None:
    memories = LedgerRepository().list_memories(status="ACTIVE")

    assert [row["memory_id"] for row in memories] == ["memory-1", "memory-3"]


def test_list_memories_memory_type_filter_works(fake_read_db: FakeReadDatabase) -> None:
    memories = LedgerRepository().list_memories(memory_type="PROJECT_GOAL_TOPIC")

    assert [row["memory_id"] for row in memories] == ["memory-2", "memory-3"]


def test_list_memories_subject_entity_id_filter_works(fake_read_db: FakeReadDatabase) -> None:
    memories = LedgerRepository().list_memories(subject_entity_id="entity-1")

    assert [row["memory_id"] for row in memories] == ["memory-1", "memory-3"]


def test_read_helpers_do_not_write(fake_read_db: FakeReadDatabase) -> None:
    LedgerRepository().get_memory("memory-1")
    LedgerRepository().list_memories(limit=1)

    assert not any(operation.startswith(("insert ", "update ", "delete ")) for operation in fake_read_db.operations)


def summary_row(
    memory_id: str,
    canonical_text: str,
    status: str,
    memory_type: str,
    subject_entity_id: str,
    created_order: int,
) -> dict:
    return {
        "memory_id": memory_id,
        "canonical_text": canonical_text,
        "subject_entity_id": subject_entity_id,
        "subject_text": "subject",
        "subject_entity_type": "person",
        "predicate_type": "TEST",
        "memory_type": memory_type,
        "modality": "FACT",
        "polarity": "POSITIVE",
        "certainty": "CERTAIN",
        "explicitness": "EXPLICIT",
        "attributed_to": "user",
        "temporal_kind": "NONE",
        "valid_from": None,
        "valid_to": None,
        "event_time": None,
        "temporal_precision": "NONE",
        "recurrence": "NONE",
        "recurrence_specifics": None,
        "status": status,
        "version": 1,
        "created_at": dt(created_order),
        "updated_at": dt(created_order),
    }


def dt(second: int) -> datetime:
    return datetime(2026, 9, 6, 1, 2, second, tzinfo=timezone.utc)
