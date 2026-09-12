"""PostgreSQL repository for the add-only Canonical Memory Ledger."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from kivi_memory.entity_resolution.repository import connect
from kivi_memory.ledger.models import AddMemoryResult, MemoryNotFoundError, StoredMemory


class LedgerRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def connect(self):
        return connect(self.database_url)

    def add_memory_record(
        self,
        *,
        memory_values: dict[str, Any],
        argument_values: list[dict[str, Any]],
        evidence_values: list[dict[str, Any]],
        event_payload: dict[str, Any],
    ) -> AddMemoryResult:
        """Insert one memory and its ledger children in a single transaction."""

        with self.connect() as conn:
            with conn.cursor() as cur:
                return insert_memory_record_in_transaction(
                    cur,
                    memory_values=memory_values,
                    argument_values=argument_values,
                    evidence_values=evidence_values,
                    event_payload=event_payload,
                )

    def lock_active_memory(self, cur, memory_id: str) -> dict[str, Any]:
        cur.execute(
            """
            SELECT memory_id::text AS memory_id, status, version
            FROM semantic_memories
            WHERE memory_id = %s
            FOR UPDATE
            """,
            (memory_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise MemoryNotFoundError(f"memory does not exist: {memory_id}")
        if row["status"] != "ACTIVE":
            raise ValueError(f"target memory is not ACTIVE: {memory_id}")
        return row

    def append_evidence_in_transaction(
        self,
        cur,
        *,
        memory_id: str,
        evidence_values: list[dict[str, Any]],
    ) -> int:
        evidence_sql = """
            INSERT INTO memory_evidence (
                memory_id,
                episode_id,
                message_id,
                source_text,
                observed_at
            )
            SELECT %s, %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1
                FROM memory_evidence
                WHERE memory_id = %s
                  AND episode_id = %s
                  AND message_id = %s
                  AND source_text = %s
            )
        """
        inserted = 0
        for evidence in evidence_values:
            cur.execute(
                evidence_sql,
                (
                    memory_id,
                    evidence["episode_id"],
                    evidence["message_id"],
                    evidence["source_text"],
                    evidence["observed_at"],
                    memory_id,
                    evidence["episode_id"],
                    evidence["message_id"],
                    evidence["source_text"],
                ),
            )
            inserted += cur.rowcount or 0
        return inserted

    def increment_memory_version_in_transaction(self, cur, *, memory_id: str) -> None:
        cur.execute(
            """
            UPDATE semantic_memories
            SET version = version + 1,
                updated_at = now()
            WHERE memory_id = %s
              AND status = 'ACTIVE'
            """,
            (memory_id,),
        )
        if cur.rowcount != 1:
            raise ValueError(f"target memory could not be reinforced: {memory_id}")

    def update_memory_status_in_transaction(self, cur, *, memory_id: str, status: str) -> None:
        cur.execute(
            """
            UPDATE semantic_memories
            SET status = %s,
                version = version + 1,
                updated_at = now()
            WHERE memory_id = %s
              AND status = 'ACTIVE'
            """,
            (status, memory_id),
        )
        if cur.rowcount != 1:
            raise ValueError(f"target memory could not be updated: {memory_id}")

    def insert_event_in_transaction(
        self,
        cur,
        *,
        memory_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        cur.execute(
            "INSERT INTO memory_events (memory_id, event_type, payload) VALUES (%s, %s, %s)",
            (memory_id, event_type, Jsonb(payload)),
        )

    def get_memory(self, memory_id: str) -> StoredMemory:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM semantic_memories WHERE memory_id = %s", (memory_id,))
                memory = cur.fetchone()
                if memory is None:
                    raise MemoryNotFoundError(f"memory does not exist: {memory_id}")

                cur.execute(
                    "SELECT * FROM memory_arguments WHERE memory_id = %s ORDER BY position",
                    (memory_id,),
                )
                arguments = cur.fetchall()

                cur.execute(
                    "SELECT * FROM memory_evidence WHERE memory_id = %s ORDER BY created_at, evidence_id",
                    (memory_id,),
                )
                evidence = cur.fetchall()

                cur.execute(
                    "SELECT * FROM memory_events WHERE memory_id = %s ORDER BY created_at, event_id",
                    (memory_id,),
                )
                events = cur.fetchall()

        return StoredMemory(
            memory=_jsonable_row(memory),
            arguments=[_jsonable_row(row) for row in arguments],
            evidence=[_jsonable_row(row) for row in evidence],
            events=[_jsonable_row(row) for row in events],
        )

    def list_memories(
        self,
        limit: int = 50,
        status: str | None = None,
        memory_type: str | None = None,
        subject_entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be at least 1")

        clauses = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if memory_type is not None:
            clauses.append("memory_type = %s")
            params.append(memory_type)
        if subject_entity_id is not None:
            clauses.append("subject_entity_id = %s")
            params.append(subject_entity_id)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        sql = f"""
            SELECT
                memory_id,
                canonical_text,
                subject_entity_id,
                subject_text,
                predicate_type,
                memory_type,
                status,
                version,
                created_at,
                updated_at
            FROM semantic_memories
            {where_sql}
            ORDER BY created_at DESC, memory_id DESC
            LIMIT %s
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return [_jsonable_row(row) for row in cur.fetchall()]


def insert_memory_record_in_transaction(
    cur,
    *,
    memory_values: dict[str, Any],
    argument_values: list[dict[str, Any]],
    evidence_values: list[dict[str, Any]],
    event_payload: dict[str, Any],
) -> AddMemoryResult:
    memory_sql = """
        INSERT INTO semantic_memories (
            canonical_text,
            subject_entity_id,
            subject_text,
            subject_entity_type,
            predicate_type,
            memory_type,
            modality,
            polarity,
            certainty,
            explicitness,
            attributed_to,
            temporal_kind,
            valid_from,
            valid_to,
            event_time,
            temporal_precision,
            recurrence,
            recurrence_specifics,
            status,
            version
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING memory_id::text AS memory_id, status, version
    """
    argument_sql = """
        INSERT INTO memory_arguments (
            memory_id,
            role,
            text,
            is_entity,
            entity_id,
            entity_type,
            position
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    evidence_sql = """
        INSERT INTO memory_evidence (
            memory_id,
            episode_id,
            message_id,
            source_text,
            observed_at
        )
        VALUES (%s, %s, %s, %s, %s)
    """
    event_sql = """
        INSERT INTO memory_events (memory_id, event_type, payload)
        VALUES (%s, %s, %s)
    """

    cur.execute(
        memory_sql,
        (
            memory_values["canonical_text"],
            memory_values["subject_entity_id"],
            memory_values["subject_text"],
            memory_values["subject_entity_type"],
            memory_values["predicate_type"],
            memory_values["memory_type"],
            memory_values["modality"],
            memory_values["polarity"],
            memory_values["certainty"],
            memory_values["explicitness"],
            memory_values["attributed_to"],
            memory_values["temporal_kind"],
            memory_values["valid_from"],
            memory_values["valid_to"],
            memory_values["event_time"],
            memory_values["temporal_precision"],
            memory_values["recurrence"],
            memory_values["recurrence_specifics"],
            "ACTIVE",
            1,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("semantic memory insert did not return a row")
    memory_id = row["memory_id"]

    for argument in argument_values:
        cur.execute(
            argument_sql,
            (
                memory_id,
                argument["role"],
                argument["text"],
                argument["is_entity"],
                argument["entity_id"],
                argument["entity_type"],
                argument["position"],
            ),
        )

    for evidence in evidence_values:
        cur.execute(
            evidence_sql,
            (
                memory_id,
                evidence["episode_id"],
                evidence["message_id"],
                evidence["source_text"],
                evidence["observed_at"],
            ),
        )

    cur.execute(event_sql, (memory_id, "MEMORY_ADDED", Jsonb(event_payload)))
    return AddMemoryResult(memory_id=memory_id, status=row["status"], version=int(row["version"]))


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if isinstance(value, (datetime, date)):
            result[key] = value.isoformat()
        elif isinstance(value, UUID):
            result[key] = str(value)
        else:
            result[key] = value
    return result
