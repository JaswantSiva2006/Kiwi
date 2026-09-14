from __future__ import annotations

import time
from typing import Any

from .embedding_service import DocumentEmbeddingService
from .models import IngestionResult


def ingest_parsed_document(
    parsed: dict[str, Any],
    *,
    repository: DocumentRagRepository | None = None,
    embedding_service: DocumentEmbeddingService | None = None,
) -> IngestionResult:
    started = time.perf_counter()
    if repository is None:
        from .repository import DocumentRagRepository

        repository = DocumentRagRepository()
    embedding_service = embedding_service or DocumentEmbeddingService()

    _validate_parsed_document(parsed)
    document = parsed["document"]
    existing = repository.find_document_by_sha256(document["sha256"])
    if existing is not None:
        total_time = time.perf_counter() - started
        return IngestionResult(
            document_id=existing["document_id"],
            chunks_indexed=0,
            duplicate=True,
            embedding_time=0.0,
            db_write_time=0.0,
            total_time=total_time,
        )

    texts = [chunk["text"] for chunk in parsed["chunks"]]
    embedding_started = time.perf_counter()
    embeddings = embedding_service.embed_documents(texts)
    embedding_time = time.perf_counter() - embedding_started

    db_started = time.perf_counter()
    repository.store_document_with_chunks(parsed, embeddings)
    db_write_time = time.perf_counter() - db_started

    return IngestionResult(
        document_id=document["document_id"],
        chunks_indexed=len(parsed["chunks"]),
        duplicate=False,
        embedding_time=embedding_time,
        db_write_time=db_write_time,
        total_time=time.perf_counter() - started,
    )


def _validate_parsed_document(parsed: dict[str, Any]) -> None:
    document = parsed.get("document") or {}
    if document.get("parse_status") != "OK":
        raise ValueError("parsed document must have parse_status OK")
    chunks = parsed.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("parsed document must contain chunks")
    required_document_fields = {"document_id", "filename", "sha256", "page_count"}
    missing_document = required_document_fields - set(document)
    if missing_document:
        raise ValueError(f"missing document fields: {sorted(missing_document)}")
    seen_ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        if chunk.get("chunk_index") != index:
            raise ValueError("chunk indexes must be contiguous and ordered")
        if not chunk.get("text", "").strip():
            raise ValueError(f"empty chunk text at index {index}")
        if chunk.get("token_count", 0) <= 0:
            raise ValueError(f"invalid token_count at index {index}")
        if chunk.get("page_start", 0) > chunk.get("page_end", 0):
            raise ValueError(f"invalid page range at index {index}")
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen_ids:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)
