"""Read-only hydration for reconciliation candidates."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Any
from uuid import UUID

from kivi_memory.entity_resolution.repository import connect
from kivi_memory.reconciliation.models import HydratedMemory, HydratedMemoryArgument


class ReconciliationRepository:
    """Fetch only the ledger fields needed by the reconciliation prompt."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def hydrate_candidates(self, memory_ids: list[str]) -> list[HydratedMemory]:
        if not memory_ids:
            return []

        sql_memories = """
            SELECT
                memory_id::text AS memory_id,
                canonical_text,
                subject_entity_id::text AS subject_entity_id,
                subject_text,
                predicate_type,
                memory_type,
                modality,
                polarity,
                certainty,
                temporal_kind,
                valid_from,
                valid_to,
                event_time,
                temporal_precision,
                recurrence,
                recurrence_specifics,
                status
            FROM semantic_memories
            WHERE memory_id = ANY(%s::uuid[])
              AND status = 'ACTIVE'
        """
        sql_arguments = """
            SELECT
                memory_id::text AS memory_id,
                role,
                text,
                is_entity,
                entity_id::text AS entity_id,
                entity_type,
                position
            FROM memory_arguments
            WHERE memory_id = ANY(%s::uuid[])
            ORDER BY memory_id, position
        """

        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_memories, (memory_ids,))
                memory_rows = cur.fetchall()
                cur.execute(sql_arguments, (memory_ids,))
                argument_rows = cur.fetchall()

        arguments_by_memory: dict[str, list[HydratedMemoryArgument]] = defaultdict(list)
        for row in argument_rows:
            arguments_by_memory[row["memory_id"]].append(
                HydratedMemoryArgument(
                    role=row["role"],
                    text=row["text"],
                    is_entity=bool(row["is_entity"]),
                    entity_id=row["entity_id"],
                    entity_type=row["entity_type"],
                    position=int(row["position"]),
                )
            )

        memories = {
            row["memory_id"]: HydratedMemory(
                memory_id=row["memory_id"],
                canonical_text=row["canonical_text"],
                subject_entity_id=row["subject_entity_id"],
                subject_text=row["subject_text"],
                predicate_type=row["predicate_type"],
                memory_type=row["memory_type"],
                modality=row["modality"],
                polarity=row["polarity"],
                certainty=row["certainty"],
                temporal_kind=row["temporal_kind"],
                valid_from=_jsonable_time(row["valid_from"]),
                valid_to=_jsonable_time(row["valid_to"]),
                event_time=_jsonable_time(row["event_time"]),
                temporal_precision=row["temporal_precision"],
                recurrence=row["recurrence"],
                recurrence_specifics=row["recurrence_specifics"],
                status=row["status"],
                arguments=arguments_by_memory.get(row["memory_id"], []),
            )
            for row in memory_rows
        }
        return [memories[memory_id] for memory_id in memory_ids if memory_id in memories]


def _jsonable_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return str(value)
