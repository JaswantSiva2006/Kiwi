"""Read-only vector retrieval over indexed canonical memories."""

from __future__ import annotations

import logging
import time

from kivi_memory.embeddings.config import (
    EMBEDDING_MODEL_NAME,
    HNSW_ITERATIVE_SCAN,
    VECTOR_SEARCH_MODE_EXACT,
    VECTOR_SEARCH_MODE_HNSW,
    get_hnsw_ef_search,
    get_vector_search_mode,
)
from kivi_memory.embeddings.generator import embed_text
from kivi_memory.embeddings.models import VectorMemoryCandidate, VectorRetrievalResult
from kivi_memory.embeddings.repository import MemoryEmbeddingRepository

logger = logging.getLogger(__name__)

MAX_VECTOR_RETRIEVAL_TOP_K = 100


def retrieve_vector_candidates(
    canonical_text: str,
    top_k: int = 20,
    repository: MemoryEmbeddingRepository | None = None,
    embedder=embed_text,
    search_mode: str | None = None,
) -> VectorRetrievalResult:
    """Embed one query and return nearest ACTIVE memory summaries."""

    if top_k < 1 or top_k > MAX_VECTOR_RETRIEVAL_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_VECTOR_RETRIEVAL_TOP_K}")
    if not canonical_text.strip():
        raise ValueError("canonical_text must not be empty")

    mode = (search_mode or get_vector_search_mode()).strip().lower()
    if mode not in {VECTOR_SEARCH_MODE_EXACT, VECTOR_SEARCH_MODE_HNSW}:
        raise ValueError("vector search mode must be exact or hnsw")

    total_started = time.perf_counter()
    embedding_started = time.perf_counter()
    query_embedding = embedder(canonical_text)
    embedding_ms = (time.perf_counter() - embedding_started) * 1000

    result = retrieve_vector_candidates_by_vector(
        query_embedding,
        top_k=top_k,
        repository=repository,
        search_mode=mode,
    )
    return VectorRetrievalResult(
        candidates=result.candidates,
        embedding_ms=embedding_ms,
        db_search_ms=result.db_search_ms,
        total_ms=(time.perf_counter() - total_started) * 1000,
        vector_search_mode=result.vector_search_mode,
        vector_top_k=top_k,
        vector_returned_count=len(result.candidates),
        hnsw_ef_search=result.hnsw_ef_search,
        hnsw_iterative_scan=result.hnsw_iterative_scan,
        hnsw_fallback_used=result.hnsw_fallback_used,
        hnsw_fallback_reason=result.hnsw_fallback_reason,
    )


def retrieve_vector_candidates_by_vector(
    query_embedding: list[float],
    top_k: int = 20,
    repository: MemoryEmbeddingRepository | None = None,
    search_mode: str | None = None,
) -> VectorRetrievalResult:
    """Retrieve by a precomputed query vector for fair exact/HNSW comparisons."""

    if top_k < 1 or top_k > MAX_VECTOR_RETRIEVAL_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_VECTOR_RETRIEVAL_TOP_K}")
    mode = (search_mode or get_vector_search_mode()).strip().lower()
    repo = repository or MemoryEmbeddingRepository()
    if mode == VECTOR_SEARCH_MODE_EXACT:
        return retrieve_vector_candidates_exact_by_vector(query_embedding, top_k, repo)
    if mode == VECTOR_SEARCH_MODE_HNSW:
        return retrieve_vector_candidates_hnsw_by_vector(query_embedding, top_k, repo)
    raise ValueError("vector search mode must be exact or hnsw")


def retrieve_vector_candidates_exact(
    canonical_text: str,
    top_k: int = 20,
    repository: MemoryEmbeddingRepository | None = None,
    embedder=embed_text,
) -> VectorRetrievalResult:
    return retrieve_vector_candidates(
        canonical_text,
        top_k=top_k,
        repository=repository,
        embedder=embedder,
        search_mode=VECTOR_SEARCH_MODE_EXACT,
    )


def retrieve_vector_candidates_hnsw(
    canonical_text: str,
    top_k: int = 20,
    repository: MemoryEmbeddingRepository | None = None,
    embedder=embed_text,
) -> VectorRetrievalResult:
    return retrieve_vector_candidates(
        canonical_text,
        top_k=top_k,
        repository=repository,
        embedder=embedder,
        search_mode=VECTOR_SEARCH_MODE_HNSW,
    )


def retrieve_vector_candidates_exact_by_vector(
    query_embedding: list[float],
    top_k: int,
    repository: MemoryEmbeddingRepository | None = None,
) -> VectorRetrievalResult:
    repo = repository or MemoryEmbeddingRepository()
    started = time.perf_counter()
    candidates = repo.retrieve_vector_candidates_exact(query_embedding, EMBEDDING_MODEL_NAME, top_k)
    db_ms = (time.perf_counter() - started) * 1000
    return _result(candidates, db_ms, VECTOR_SEARCH_MODE_EXACT, top_k)


def retrieve_vector_candidates_hnsw_by_vector(
    query_embedding: list[float],
    top_k: int,
    repository: MemoryEmbeddingRepository | None = None,
) -> VectorRetrievalResult:
    repo = repository or MemoryEmbeddingRepository()
    ef_search = get_hnsw_ef_search(top_k)
    try:
        if hasattr(repo, "hnsw_index_exists") and not repo.hnsw_index_exists():
            raise RuntimeError("expected HNSW index is missing")
        started = time.perf_counter()
        candidates = repo.retrieve_vector_candidates_hnsw(query_embedding, EMBEDDING_MODEL_NAME, top_k, ef_search)
        db_ms = (time.perf_counter() - started) * 1000
        if len(candidates) < top_k and _has_enough_active_embeddings(repo, top_k):
            raise RuntimeError(f"HNSW returned {len(candidates)} candidates for top_k={top_k}")
        return _result(candidates, db_ms, VECTOR_SEARCH_MODE_HNSW, top_k, ef_search=ef_search)
    except Exception as exc:
        logger.warning("HNSW retrieval failed; falling back to exact search: %s", exc)
        exact = retrieve_vector_candidates_exact_by_vector(query_embedding, top_k, repo)
        return VectorRetrievalResult(
            candidates=exact.candidates,
            embedding_ms=0.0,
            db_search_ms=exact.db_search_ms,
            total_ms=exact.total_ms,
            vector_search_mode=VECTOR_SEARCH_MODE_EXACT,
            vector_top_k=top_k,
            vector_returned_count=len(exact.candidates),
            hnsw_ef_search=ef_search,
            hnsw_iterative_scan=HNSW_ITERATIVE_SCAN,
            hnsw_fallback_used=True,
            hnsw_fallback_reason=str(exc),
        )


def _has_enough_active_embeddings(repo: MemoryEmbeddingRepository, top_k: int) -> bool:
    if not hasattr(repo, "count_active_embeddings"):
        return False
    return repo.count_active_embeddings(EMBEDDING_MODEL_NAME) >= top_k


def _result(
    candidates: list[VectorMemoryCandidate],
    db_ms: float,
    mode: str,
    top_k: int,
    *,
    ef_search: int | None = None,
) -> VectorRetrievalResult:
    return VectorRetrievalResult(
        candidates=candidates,
        embedding_ms=0.0,
        db_search_ms=db_ms,
        total_ms=db_ms,
        vector_search_mode=mode,
        vector_top_k=top_k,
        vector_returned_count=len(candidates),
        hnsw_ef_search=ef_search,
        hnsw_iterative_scan=HNSW_ITERATIVE_SCAN if mode == VECTOR_SEARCH_MODE_HNSW else None,
        hnsw_fallback_used=False,
        hnsw_fallback_reason=None,
    )
