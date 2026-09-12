"""Return models for memory embedding indexing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexMemoryResult:
    memory_id: str
    embedding_model: str
    indexed: bool
    skipped: bool
    embedding_text_hash: str


@dataclass(frozen=True)
class BackfillResult:
    total_active_memories: int
    indexed: int
    skipped_current: int
    failed: int


@dataclass(frozen=True)
class VectorMemoryCandidate:
    memory_id: str
    canonical_text: str
    vector_similarity: float
    subject_entity_id: str | None
    predicate_type: str
    memory_type: str
    status: str


@dataclass(frozen=True)
class VectorRetrievalResult:
    candidates: list[VectorMemoryCandidate]
    embedding_ms: float
    db_search_ms: float
    total_ms: float
    vector_search_mode: str = "exact"
    vector_top_k: int = 20
    vector_returned_count: int = 0
    hnsw_ef_search: int | None = None
    hnsw_iterative_scan: str | None = None
    hnsw_fallback_used: bool = False
    hnsw_fallback_reason: str | None = None
