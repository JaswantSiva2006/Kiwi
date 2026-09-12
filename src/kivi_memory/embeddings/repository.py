"""PostgreSQL storage for derived memory embeddings."""

from __future__ import annotations

from typing import Any

from kivi_memory.embeddings.config import HNSW_INDEX_NAME, HNSW_ITERATIVE_SCAN
from kivi_memory.entity_resolution.repository import connect
from kivi_memory.embeddings.models import VectorMemoryCandidate


class MemoryEmbeddingRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def get_memory_canonical_text(self, memory_id: str) -> str:
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT canonical_text FROM semantic_memories WHERE memory_id = %s", (memory_id,))
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"memory does not exist: {memory_id}")
                return row["canonical_text"]

    def get_embedding_text_hash(self, memory_id: str, embedding_model: str) -> str | None:
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT embedding_text_hash
                    FROM memory_embeddings
                    WHERE memory_id = %s AND embedding_model = %s
                    """,
                    (memory_id, embedding_model),
                )
                row = cur.fetchone()
                return None if row is None else row["embedding_text_hash"]

    def store_memory_embedding(
        self,
        memory_id: str,
        embedding: list[float],
        embedding_model: str,
        embedding_text_hash: str,
    ) -> None:
        sql = """
            INSERT INTO memory_embeddings (
                memory_id,
                embedding_model,
                embedding,
                embedding_text_hash
            )
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (memory_id, embedding_model)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_text_hash = EXCLUDED.embedding_text_hash,
                updated_at = now()
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (memory_id, embedding_model, _vector_literal(embedding), embedding_text_hash))

    def list_active_memory_embedding_status(self, embedding_model: str, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")

        limit_sql = "LIMIT %s" if limit is not None else ""
        params: list[Any] = [embedding_model]
        if limit is not None:
            params.append(limit)
        sql = f"""
            SELECT
                m.memory_id::text AS memory_id,
                m.canonical_text,
                e.embedding_text_hash
            FROM semantic_memories m
            LEFT JOIN memory_embeddings e
              ON e.memory_id = m.memory_id
             AND e.embedding_model = %s
            WHERE m.status = 'ACTIVE'
            ORDER BY m.created_at ASC, m.memory_id ASC
            {limit_sql}
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return [dict(row) for row in cur.fetchall()]

    def retrieve_vector_candidates(
        self,
        query_embedding: list[float],
        embedding_model: str,
        top_k: int,
    ) -> list[VectorMemoryCandidate]:
        return self.retrieve_vector_candidates_exact(query_embedding, embedding_model, top_k)

    def retrieve_vector_candidates_exact(
        self,
        query_embedding: list[float],
        embedding_model: str,
        top_k: int,
    ) -> list[VectorMemoryCandidate]:
        sql = """
            SELECT
                sm.memory_id::text AS memory_id,
                sm.canonical_text,
                1 - (me.embedding <=> %s::vector) AS vector_similarity,
                sm.subject_entity_id::text AS subject_entity_id,
                sm.predicate_type,
                sm.memory_type,
                sm.status
            FROM memory_embeddings me
            JOIN semantic_memories sm ON sm.memory_id = me.memory_id
            WHERE sm.status = 'ACTIVE'
              AND me.embedding_model = %s
            ORDER BY me.embedding <=> %s::vector
            LIMIT %s
        """
        vector = _vector_literal(query_embedding)
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute(sql, (vector, embedding_model, vector, top_k))
                return _rows_to_candidates(cur.fetchall())

    def retrieve_vector_candidates_hnsw(
        self,
        query_embedding: list[float],
        embedding_model: str,
        top_k: int,
        ef_search: int,
    ) -> list[VectorMemoryCandidate]:
        sql = """
            SELECT
                sm.memory_id::text AS memory_id,
                sm.canonical_text,
                1 - (me.embedding <=> %s::vector) AS vector_similarity,
                sm.subject_entity_id::text AS subject_entity_id,
                sm.predicate_type,
                sm.memory_type,
                sm.status
            FROM memory_embeddings me
            JOIN semantic_memories sm ON sm.memory_id = me.memory_id
            WHERE sm.status = 'ACTIVE'
              AND me.embedding_model = %s
            ORDER BY me.embedding <=> %s::vector
            LIMIT %s
        """
        vector = _vector_literal(query_embedding)
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
                cur.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", (HNSW_ITERATIVE_SCAN,))
                cur.execute(sql, (vector, embedding_model, vector, top_k))
                return _rows_to_candidates(cur.fetchall())

    def count_active_embeddings(self, embedding_model: str) -> int:
        sql = """
            SELECT count(*) AS count
            FROM memory_embeddings me
            JOIN semantic_memories sm ON sm.memory_id = me.memory_id
            WHERE sm.status = 'ACTIVE'
              AND me.embedding_model = %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (embedding_model,))
                row = cur.fetchone()
                return int(row["count"])

    def hnsw_index_exists(self) -> bool:
        sql = """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'memory_embeddings'
              AND indexname = %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (HNSW_INDEX_NAME,))
                return cur.fetchone() is not None

    def vector_health(self, embedding_model: str) -> dict[str, Any]:
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                version_row = cur.fetchone()
                cur.execute(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'memory_embeddings'
                    ORDER BY indexname
                    """
                )
                indexes = cur.fetchall()
                cur.execute("SELECT current_setting('hnsw.ef_search', true) AS current_ef_search")
                ef_search = cur.fetchone()["current_ef_search"]
                cur.execute("SELECT count(*) AS count FROM memory_embeddings WHERE embedding_model = %s", (embedding_model,))
                embedding_count = int(cur.fetchone()["count"])
                cur.execute("SELECT count(*) AS count FROM semantic_memories WHERE status = 'ACTIVE'")
                active_count = int(cur.fetchone()["count"])
        hnsw_index = next((row for row in indexes if row["indexname"] == HNSW_INDEX_NAME), None)
        return {
            "pgvector_version": None if version_row is None else version_row["extversion"],
            "current_embedding_model": embedding_model,
            "hnsw_index_exists": hnsw_index is not None,
            "hnsw_index_definition": None if hnsw_index is None else hnsw_index["indexdef"],
            "indexes": [dict(row) for row in indexes],
            "embedding_count_for_model": embedding_count,
            "active_memory_count": active_count,
            "current_ef_search": ef_search,
        }

    def explain_hnsw_query(
        self,
        query_embedding: list[float],
        embedding_model: str,
        top_k: int,
        ef_search: int,
    ) -> dict[str, Any]:
        sql = """
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT
                sm.memory_id::text AS memory_id,
                sm.canonical_text,
                1 - (me.embedding <=> %s::vector) AS vector_similarity,
                sm.subject_entity_id::text AS subject_entity_id,
                sm.predicate_type,
                sm.memory_type,
                sm.status
            FROM memory_embeddings me
            JOIN semantic_memories sm ON sm.memory_id = me.memory_id
            WHERE sm.status = 'ACTIVE'
              AND me.embedding_model = %s
            ORDER BY me.embedding <=> %s::vector
            LIMIT %s
        """
        vector = _vector_literal(query_embedding)
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
                cur.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", (HNSW_ITERATIVE_SCAN,))
                cur.execute(sql, (vector, embedding_model, vector, top_k))
                plan_lines = [row["QUERY PLAN"] for row in cur.fetchall()]
        plan = "\n".join(plan_lines)
        return {
            "plan": plan,
            "hnsw_index_used": HNSW_INDEX_NAME in plan,
            "rows_returned_estimate": _rows_from_plan(plan),
        }


def _rows_to_candidates(rows: list[dict[str, Any]]) -> list[VectorMemoryCandidate]:
    return [
        VectorMemoryCandidate(
            memory_id=row["memory_id"],
            canonical_text=row["canonical_text"],
            vector_similarity=float(row["vector_similarity"]),
            subject_entity_id=row["subject_entity_id"],
            predicate_type=row["predicate_type"],
            memory_type=row["memory_type"],
            status=row["status"],
        )
        for row in rows
    ]


def _rows_from_plan(plan: str) -> int | None:
    marker = "actual time="
    if marker not in plan:
        return None
    for line in plan.splitlines():
        if "Limit" in line and "rows=" in line:
            try:
                return int(line.split("rows=")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"
