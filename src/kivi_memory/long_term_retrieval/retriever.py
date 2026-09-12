"""Public read-only long-term memory retrieval API."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kivi_memory.embeddings import retrieve_vector_candidates
from kivi_memory.long_term_retrieval.config import clamp_branch_k, clamp_top_k
from kivi_memory.long_term_retrieval.fusion import fuse_branch_candidates
from kivi_memory.long_term_retrieval.models import BranchCandidate, RetrievalDiagnostics, RetrievalResult
from kivi_memory.long_term_retrieval.repository import LongTermRetrievalRepository

SOURCE_VECTOR = "VECTOR"
SOURCE_LEXICAL = "LEXICAL"
SOURCE_STRUCTURED = "STRUCTURED"
SOURCE_GRAPH = "GRAPH"


def retrieve_long_term_memories(
    query_text: str,
    top_k: int = 12,
    branch_k: int = 30,
    repository: LongTermRetrievalRepository | None = None,
) -> RetrievalResult:
    """Retrieve a compact, grounded bundle of ACTIVE long-term memories."""

    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    top_k = clamp_top_k(top_k)
    branch_k = clamp_branch_k(branch_k)
    repo = repository or LongTermRetrievalRepository()
    total_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as executor:
        vector_future = executor.submit(_vector_branch, query_text, branch_k)
        lexical_future = executor.submit(_timed, repo.retrieve_lexical_candidates, query_text, branch_k)
        entity_future = executor.submit(_timed, repo.match_query_entities, query_text)

        vector_candidates, vector_ms, vector_meta = vector_future.result()
        lexical_candidates, lexical_ms = lexical_future.result()
        resolved_entities, entity_match_ms = entity_future.result()

    entity_ids = [entity.entity_id for entity in resolved_entities]
    with ThreadPoolExecutor(max_workers=2) as executor:
        structured_future = executor.submit(_timed, repo.retrieve_structured_candidates, entity_ids, branch_k)
        graph_future = executor.submit(_timed, repo.retrieve_graph_candidates, entity_ids, branch_k)
        structured_candidates, structured_ms = structured_future.result()
        graph_candidates, graph_ms = graph_future.result()

    fusion_started = time.perf_counter()
    branch_candidates = {
        SOURCE_VECTOR: vector_candidates,
        SOURCE_LEXICAL: lexical_candidates,
        SOURCE_STRUCTURED: structured_candidates,
        SOURCE_GRAPH: graph_candidates,
    }
    fused = fuse_branch_candidates(branch_candidates, top_k=top_k)
    fusion_ms = _elapsed_ms(fusion_started)

    hydrate_started = time.perf_counter()
    memories = repo.hydrate_memories(fused)
    hydrate_ms = _elapsed_ms(hydrate_started)

    diagnostics = RetrievalDiagnostics(
        vector_ms=vector_ms,
        lexical_ms=lexical_ms,
        entity_match_ms=entity_match_ms,
        structured_ms=structured_ms,
        graph_ms=graph_ms,
        fusion_ms=fusion_ms,
        hydrate_ms=hydrate_ms,
        total_ms=_elapsed_ms(total_started),
        vector_count=len(vector_candidates),
        lexical_count=len(lexical_candidates),
        structured_count=len(structured_candidates),
        graph_count=len(graph_candidates),
        union_unique_count=len({candidate.memory_id for candidates in branch_candidates.values() for candidate in candidates}),
        hnsw_used=vector_meta["hnsw_used"],
        hnsw_fallback_used=vector_meta["hnsw_fallback_used"],
    )
    return RetrievalResult(
        query=query_text,
        resolved_query_entities=resolved_entities,
        memories=memories,
        diagnostics=diagnostics,
        branch_candidates=branch_candidates,
    )


def _vector_branch(query_text: str, branch_k: int) -> tuple[list[BranchCandidate], float, dict[str, bool]]:
    started = time.perf_counter()
    result = retrieve_vector_candidates(query_text, top_k=branch_k)
    candidates = [
        BranchCandidate(
            memory_id=candidate.memory_id,
            source=SOURCE_VECTOR,
            rank=index,
            raw_score=candidate.vector_similarity,
            diagnostics={"vector_similarity": candidate.vector_similarity},
        )
        for index, candidate in enumerate(result.candidates, start=1)
    ]
    return (
        candidates,
        _elapsed_ms(started),
        {
            "hnsw_used": result.vector_search_mode == "hnsw" and not result.hnsw_fallback_used,
            "hnsw_fallback_used": result.hnsw_fallback_used,
        },
    )


def _timed(function, *args) -> tuple[Any, float]:
    started = time.perf_counter()
    return function(*args), _elapsed_ms(started)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
