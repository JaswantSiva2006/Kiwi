"""PostgreSQL access for the derived memory-entity graph projection."""

from __future__ import annotations

from typing import Any

from kivi_memory.entity_resolution.repository import connect
from kivi_memory.ledger.models import MemoryNotFoundError

from kivi_memory.graph.models import GraphSourceArgument, GraphSourceMemory, MemoryEntityLink


class GraphRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def connect(self):
        return connect(self.database_url)

    def get_memory_graph_source(self, memory_id: str) -> GraphSourceMemory:
        """Read the committed ledger rows needed for deterministic projection."""

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        memory_id::text AS memory_id,
                        predicate_type,
                        subject_entity_id::text AS subject_entity_id
                    FROM semantic_memories
                    WHERE memory_id = %s
                    """,
                    (memory_id,),
                )
                memory = cur.fetchone()
                if memory is None:
                    raise MemoryNotFoundError(f"memory does not exist: {memory_id}")

                cur.execute(
                    """
                    SELECT
                        role,
                        is_entity,
                        entity_id::text AS entity_id,
                        position
                    FROM memory_arguments
                    WHERE memory_id = %s
                    ORDER BY position
                    """,
                    (memory_id,),
                )
                arguments = [
                    GraphSourceArgument(
                        role=row["role"],
                        is_entity=bool(row["is_entity"]),
                        entity_id=row["entity_id"],
                        position=int(row["position"]),
                    )
                    for row in cur.fetchall()
                ]

        return GraphSourceMemory(
            memory_id=memory["memory_id"],
            predicate_type=memory["predicate_type"],
            subject_entity_id=memory["subject_entity_id"],
            arguments=arguments,
        )

    def insert_memory_entity_links(self, links: list[MemoryEntityLink]) -> list[MemoryEntityLink]:
        if not links:
            return []

        sql = """
            INSERT INTO memory_entity_links (
                memory_id,
                entity_id,
                link_type,
                argument_role,
                predicate_type,
                position
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING
                memory_id::text AS memory_id,
                entity_id::text AS entity_id,
                link_type,
                argument_role,
                predicate_type,
                position
        """
        inserted = []
        with self.connect() as conn:
            with conn.cursor() as cur:
                for link in links:
                    cur.execute(
                        sql,
                        (
                            link.memory_id,
                            link.entity_id,
                            link.link_type,
                            link.argument_role,
                            link.predicate_type,
                            link.position,
                        ),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        inserted.append(_row_to_link(row))
        return inserted

    def list_active_memory_ids(self, limit: int | None = None) -> list[str]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        params: tuple[Any, ...]
        if limit is None:
            sql = """
                SELECT memory_id::text AS memory_id
                FROM semantic_memories
                WHERE status = 'ACTIVE'
                ORDER BY created_at ASC, memory_id ASC
            """
            params = ()
        else:
            sql = """
                SELECT memory_id::text AS memory_id
                FROM semantic_memories
                WHERE status = 'ACTIVE'
                ORDER BY created_at ASC, memory_id ASC
                LIMIT %s
            """
            params = (limit,)

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [row["memory_id"] for row in cur.fetchall()]

    def list_links_for_memory(self, memory_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        memory_id::text AS memory_id,
                        entity_id::text AS entity_id,
                        link_type,
                        argument_role,
                        predicate_type,
                        position,
                        created_at
                    FROM memory_entity_links
                    WHERE memory_id = %s
                    ORDER BY
                        CASE link_type WHEN 'SUBJECT' THEN 0 ELSE 1 END,
                        position NULLS FIRST,
                        entity_id
                    """,
                    (memory_id,),
                )
                return [dict(row) for row in cur.fetchall()]


def _row_to_link(row: dict[str, Any]) -> MemoryEntityLink:
    return MemoryEntityLink(
        memory_id=row["memory_id"],
        entity_id=row["entity_id"],
        link_type=row["link_type"],
        argument_role=row["argument_role"],
        predicate_type=row["predicate_type"],
        position=row["position"],
    )
