from __future__ import annotations

import json
from typing import Any

from .config import DOCUMENT_HNSW_INDEX_NAME, get_document_hnsw_ef_search
from .database import connect
from .models import DocumentChunkSearchResult


class DocumentRagRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def find_document_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        sql = """
            SELECT document_id, filename, sha256, page_count, status
            FROM document_rag.documents
            WHERE sha256 = %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (sha256,))
                row = cur.fetchone()
                return None if row is None else dict(row)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        sql = """
            SELECT document_id, filename, sha256, mime_type, page_count, status, created_at, metadata
            FROM document_rag.documents
            WHERE document_id = %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                row = cur.fetchone()
                return None if row is None else dict(row)

    def store_document_with_chunks(
        self,
        parsed: dict[str, Any],
        embeddings: list[list[float]],
    ) -> None:
        document = parsed["document"]
        chunks = parsed["chunks"]
        if len(chunks) != len(embeddings):
            raise ValueError("chunk and embedding counts differ")

        with connect(self.database_url) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO document_rag.documents (
                            document_id,
                            filename,
                            sha256,
                            mime_type,
                            page_count,
                            metadata,
                            status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'EMBEDDING')
                        """,
                        (
                            document["document_id"],
                            document["filename"],
                            document["sha256"],
                            "application/pdf",
                            document["page_count"],
                            json.dumps({"parser": document.get("parser"), "parse_status": document.get("parse_status")}),
                        ),
                    )
                    rows = [_chunk_row(chunk, embeddings[index]) for index, chunk in enumerate(chunks)]
                    cur.executemany(
                        """
                        INSERT INTO document_rag.chunks (
                            chunk_id,
                            document_id,
                            chunk_index,
                            text,
                            section_title,
                            section_path,
                            page_start,
                            page_end,
                            token_count,
                            embedding,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::vector, %s::jsonb)
                        """,
                        rows,
                    )
                    cur.execute(
                        """
                        UPDATE document_rag.documents
                        SET status = 'READY'
                        WHERE document_id = %s
                        """,
                        (document["document_id"],),
                    )
            except Exception:
                conn.rollback()
                self._mark_failed_if_exists(document["document_id"])
                raise

    def search_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 6,
        document_ids: list[str] | None = None,
        ef_search: int | None = None,
    ) -> list[DocumentChunkSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        params: list[Any] = [_vector_literal(query_embedding)]
        filter_sql = ""
        if document_ids:
            filter_sql = "AND c.document_id = ANY(%s)"
            params.append(document_ids)
        params.extend([_vector_literal(query_embedding), top_k])
        sql = f"""
            SELECT
                c.chunk_id,
                c.document_id,
                d.filename,
                c.text,
                c.section_title,
                c.section_path,
                c.page_start,
                c.page_end,
                1 - (c.embedding <=> %s::vector) AS similarity_score
            FROM document_rag.chunks c
            JOIN document_rag.documents d ON d.document_id = c.document_id
            WHERE d.status = 'READY'
              {filter_sql}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search or get_document_hnsw_ef_search(top_k)),))
                cur.execute(sql, tuple(params))
                return [_row_to_search_result(row) for row in cur.fetchall()]

    def hnsw_index_exists(self) -> bool:
        sql = """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = 'document_rag'
              AND tablename = 'chunks'
              AND indexname = %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (DOCUMENT_HNSW_INDEX_NAME,))
                return cur.fetchone() is not None

    def ready_document_count(self) -> int:
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS count FROM document_rag.documents WHERE status = 'READY'")
                return int(cur.fetchone()["count"])

    def explain_search(
        self,
        query_embedding: list[float],
        top_k: int = 6,
        ef_search: int | None = None,
    ) -> dict[str, Any]:
        vector = _vector_literal(query_embedding)
        sql = """
            EXPLAIN (ANALYZE, BUFFERS)
            SELECT c.chunk_id
            FROM document_rag.chunks c
            JOIN document_rag.documents d ON d.document_id = c.document_id
            WHERE d.status = 'READY'
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search or get_document_hnsw_ef_search(top_k)),))
                cur.execute(sql, (vector, top_k))
                plan = "\n".join(row["QUERY PLAN"] for row in cur.fetchall())
        return {"plan": plan, "hnsw_index_used": DOCUMENT_HNSW_INDEX_NAME in plan}

    def _mark_failed_if_exists(self, document_id: str) -> None:
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE document_rag.documents
                    SET status = 'FAILED'
                    WHERE document_id = %s
                    """,
                    (document_id,),
                )


def _chunk_row(chunk: dict[str, Any], embedding: list[float]) -> tuple[Any, ...]:
    return (
        chunk["chunk_id"],
        chunk["document_id"],
        chunk["chunk_index"],
        chunk["text"],
        chunk["section_title"],
        json.dumps(chunk["section_path"]),
        chunk["page_start"],
        chunk["page_end"],
        chunk["token_count"],
        _vector_literal(embedding),
        json.dumps(chunk["metadata"]),
    )


def _row_to_search_result(row: dict[str, Any]) -> DocumentChunkSearchResult:
    section_path = row["section_path"]
    if isinstance(section_path, str):
        section_path = json.loads(section_path)
    return DocumentChunkSearchResult(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        filename=row["filename"],
        text=row["text"],
        section_title=row["section_title"],
        section_path=list(section_path or []),
        page_start=int(row["page_start"]),
        page_end=int(row["page_end"]),
        similarity_score=float(row["similarity_score"]),
    )


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"
