"""High-level memory embedding indexing."""

from __future__ import annotations

import hashlib

from kivi_memory.embeddings.config import EMBEDDING_MODEL_NAME
from kivi_memory.embeddings.generator import embed_text, embed_texts
from kivi_memory.embeddings.models import BackfillResult, IndexMemoryResult
from kivi_memory.embeddings.repository import MemoryEmbeddingRepository


def canonical_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_memory(
    memory_id: str,
    repository: MemoryEmbeddingRepository | None = None,
    embedder=embed_text,
) -> IndexMemoryResult:
    repo = repository or MemoryEmbeddingRepository()
    canonical_text = repo.get_memory_canonical_text(memory_id)
    text_hash = canonical_text_hash(canonical_text)
    current_hash = repo.get_embedding_text_hash(memory_id, EMBEDDING_MODEL_NAME)
    if current_hash == text_hash:
        return IndexMemoryResult(
            memory_id=memory_id,
            embedding_model=EMBEDDING_MODEL_NAME,
            indexed=False,
            skipped=True,
            embedding_text_hash=text_hash,
        )

    embedding = embedder(canonical_text)
    repo.store_memory_embedding(memory_id, embedding, EMBEDDING_MODEL_NAME, text_hash)
    return IndexMemoryResult(
        memory_id=memory_id,
        embedding_model=EMBEDDING_MODEL_NAME,
        indexed=True,
        skipped=False,
        embedding_text_hash=text_hash,
    )


def backfill_active_memory_embeddings(
    batch_size: int = 32,
    repository: MemoryEmbeddingRepository | None = None,
    embedder=embed_texts,
) -> BackfillResult:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    repo = repository or MemoryEmbeddingRepository()
    rows = repo.list_active_memory_embedding_status(EMBEDDING_MODEL_NAME)
    stale_rows = [
        {**row, "current_hash": canonical_text_hash(row["canonical_text"])}
        for row in rows
        if row["embedding_text_hash"] != canonical_text_hash(row["canonical_text"])
    ]

    indexed = 0
    failed = 0
    for start in range(0, len(stale_rows), batch_size):
        batch = stale_rows[start : start + batch_size]
        try:
            embeddings = embedder([row["canonical_text"] for row in batch])
            for row, embedding in zip(batch, embeddings, strict=True):
                repo.store_memory_embedding(
                    row["memory_id"],
                    embedding,
                    EMBEDDING_MODEL_NAME,
                    row["current_hash"],
                )
                indexed += 1
        except Exception:
            failed += len(batch)

    return BackfillResult(
        total_active_memories=len(rows),
        indexed=indexed,
        skipped_current=len(rows) - len(stale_rows),
        failed=failed,
    )
