"""Pure in-memory merge and rerank for retrieval candidates."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from typing import Any

from kivi_memory.retrieval.config import DEFAULT_GRAPH_RERANK_WEIGHT, get_graph_rerank_weight
from kivi_memory.retrieval.models import (
    GraphMemoryCandidate,
    GraphSignals,
    MergedMemoryCandidate,
    StructuredMemoryCandidate,
    StructuredSignals,
)

SOURCE_VECTOR = "VECTOR"
SOURCE_STRUCTURED = "STRUCTURED"
SOURCE_GRAPH = "GRAPH"

VECTOR_SIMILARITY_WEIGHT = 1.0
SAME_SUBJECT_BOOST = 0.25
SHARED_ENTITY_BOOST = 0.20
MAX_SHARED_ENTITY_BOOST = 0.60
SAME_PREDICATE_BOOST = 0.15
SAME_MEMORY_TYPE_BOOST = 0.05
MAX_FINAL_TOP_K = 100


def merge_and_rerank_candidates(
    vector_candidates,
    structured_candidates,
    graph_candidates=None,
    final_top_k: int = 8,
) -> list[MergedMemoryCandidate]:
    """Union candidates by memory_id and return deterministic top-ranked results."""

    if final_top_k < 1 or final_top_k > MAX_FINAL_TOP_K:
        raise ValueError(f"final_top_k must be between 1 and {MAX_FINAL_TOP_K}")

    merged: dict[str, MergedMemoryCandidate] = {}
    for candidate in vector_candidates or []:
        normalized = _from_vector_candidate(candidate)
        _merge_candidate(merged, normalized)

    for candidate in structured_candidates or []:
        normalized = _from_structured_candidate(candidate)
        _merge_candidate(merged, normalized)

    for candidate in graph_candidates or []:
        normalized = _from_graph_candidate(candidate)
        _merge_candidate(merged, normalized)

    graph_weight = get_graph_rerank_weight()
    scored = [
        replace(candidate, rerank_score=_rerank_score(candidate, graph_weight=graph_weight))
        for candidate in merged.values()
    ]
    return sorted(scored, key=_sort_key, reverse=True)[:final_top_k]


def _merge_candidate(
    merged: dict[str, MergedMemoryCandidate],
    incoming: MergedMemoryCandidate,
) -> None:
    existing = merged.get(incoming.memory_id)
    if existing is None:
        merged[incoming.memory_id] = incoming
        return

    merged[incoming.memory_id] = MergedMemoryCandidate(
        memory_id=existing.memory_id,
        canonical_text=existing.canonical_text or incoming.canonical_text,
        vector_similarity=_merge_vector_similarity(existing.vector_similarity, incoming.vector_similarity),
        structured_signals=_merge_signals(existing.structured_signals, incoming.structured_signals),
        graph_signals=_merge_graph_signals(existing.graph_signals, incoming.graph_signals),
        retrieval_sources=_merge_sources(existing.retrieval_sources, incoming.retrieval_sources),
        subject_entity_id=existing.subject_entity_id or incoming.subject_entity_id,
        predicate_type=existing.predicate_type or incoming.predicate_type,
        memory_type=existing.memory_type or incoming.memory_type,
        status=existing.status or incoming.status,
    )


def _rerank_score(candidate: MergedMemoryCandidate, graph_weight: float = DEFAULT_GRAPH_RERANK_WEIGHT) -> float:
    signals = candidate.structured_signals
    score = (candidate.vector_similarity or 0.0) * VECTOR_SIMILARITY_WEIGHT
    if signals.same_subject:
        score += SAME_SUBJECT_BOOST
    score += min(signals.shared_entity_count * SHARED_ENTITY_BOOST, MAX_SHARED_ENTITY_BOOST)
    if signals.same_predicate:
        score += SAME_PREDICATE_BOOST
    if signals.same_memory_type:
        score += SAME_MEMORY_TYPE_BOOST
    score += graph_weight * max(min(candidate.graph_signals.graph_score, 1.0), 0.0)
    return score


def _merge_vector_similarity(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _sort_key(candidate: MergedMemoryCandidate) -> tuple[float, float, float, int, str]:
    return (
        candidate.rerank_score,
        candidate.vector_similarity or 0.0,
        candidate.graph_signals.graph_score,
        len(candidate.retrieval_sources),
        candidate.memory_id,
    )


def _from_vector_candidate(candidate: Any) -> MergedMemoryCandidate:
    data = _as_mapping(candidate)
    memory_id = _memory_id(data)
    return MergedMemoryCandidate(
        memory_id=memory_id,
        canonical_text=str(data.get("canonical_text") or ""),
        vector_similarity=float(data["vector_similarity"]) if data.get("vector_similarity") is not None else None,
        structured_signals=StructuredSignals(),
        graph_signals=GraphSignals(),
        retrieval_sources=[SOURCE_VECTOR],
        subject_entity_id=data.get("subject_entity_id"),
        predicate_type=data.get("predicate_type"),
        memory_type=data.get("memory_type"),
        status=data.get("status"),
    )


def _from_structured_candidate(candidate: Any) -> MergedMemoryCandidate:
    data = _as_mapping(candidate)
    memory_id = _memory_id(data)
    signals = _signals(data.get("structured_signals") or data)
    return MergedMemoryCandidate(
        memory_id=memory_id,
        canonical_text=str(data.get("canonical_text") or ""),
        vector_similarity=float(data["vector_similarity"]) if data.get("vector_similarity") is not None else None,
        structured_signals=signals,
        graph_signals=GraphSignals(),
        retrieval_sources=[SOURCE_STRUCTURED],
        subject_entity_id=data.get("subject_entity_id"),
        predicate_type=data.get("predicate_type"),
        memory_type=data.get("memory_type"),
        status=data.get("status"),
    )


def _from_graph_candidate(candidate: Any) -> MergedMemoryCandidate:
    data = _as_mapping(candidate)
    memory_id = _memory_id(data)
    graph_signals = _graph_signals(data.get("graph_signals") or data)
    return MergedMemoryCandidate(
        memory_id=memory_id,
        canonical_text=str(data.get("canonical_text") or ""),
        vector_similarity=float(data["vector_similarity"]) if data.get("vector_similarity") is not None else None,
        structured_signals=StructuredSignals(),
        graph_signals=graph_signals,
        retrieval_sources=[SOURCE_GRAPH],
        subject_entity_id=data.get("subject_entity_id"),
        predicate_type=data.get("predicate_type"),
        memory_type=data.get("memory_type"),
        status=data.get("status"),
    )


def _signals(value: Any) -> StructuredSignals:
    if isinstance(value, StructuredSignals):
        return value
    if isinstance(value, StructuredMemoryCandidate):
        return value.structured_signals
    data = _as_mapping(value)
    return StructuredSignals(
        same_subject=bool(data.get("same_subject", False)),
        shared_entity_count=max(int(data.get("shared_entity_count", 0) or 0), 0),
        same_predicate=bool(data.get("same_predicate", False)),
        same_memory_type=bool(data.get("same_memory_type", False)),
    )


def _graph_signals(value: Any) -> GraphSignals:
    if isinstance(value, GraphSignals):
        return value
    if isinstance(value, GraphMemoryCandidate):
        return value.graph_signals
    data = _as_mapping(value)
    return GraphSignals(
        graph_distance=data.get("graph_distance"),
        graph_score=max(min(float(data.get("graph_score", 0.0) or 0.0), 1.0), 0.0),
        direct_seed_count=max(int(data.get("direct_seed_count", 0) or 0), 0),
        bridge_path_count=max(int(data.get("bridge_path_count", 0) or 0), 0),
    )


def _merge_signals(left: StructuredSignals, right: StructuredSignals) -> StructuredSignals:
    return StructuredSignals(
        same_subject=left.same_subject or right.same_subject,
        shared_entity_count=max(left.shared_entity_count, right.shared_entity_count),
        same_predicate=left.same_predicate or right.same_predicate,
        same_memory_type=left.same_memory_type or right.same_memory_type,
    )


def _merge_graph_signals(left: GraphSignals, right: GraphSignals) -> GraphSignals:
    if left.graph_distance == "DIRECT" or right.graph_distance == "DIRECT":
        return GraphSignals(
            graph_distance="DIRECT",
            graph_score=1.0,
            direct_seed_count=max(left.direct_seed_count, right.direct_seed_count),
            bridge_path_count=max(left.bridge_path_count, right.bridge_path_count),
        )
    if left.graph_distance or right.graph_distance:
        return GraphSignals(
            graph_distance=left.graph_distance or right.graph_distance,
            graph_score=max(left.graph_score, right.graph_score),
            direct_seed_count=max(left.direct_seed_count, right.direct_seed_count),
            bridge_path_count=max(left.bridge_path_count, right.bridge_path_count),
        )
    return GraphSignals()


def _merge_sources(left: list[str], right: list[str]) -> list[str]:
    sources = []
    for source in [SOURCE_VECTOR, SOURCE_STRUCTURED, SOURCE_GRAPH]:
        if source in left or source in right:
            sources.append(source)
    return sources


def _memory_id(data: dict[str, Any]) -> str:
    memory_id = str(data.get("memory_id") or "").strip()
    if not memory_id:
        raise ValueError("candidate memory_id must not be empty")
    return memory_id


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"candidate must be a dict or dataclass, got {type(value).__name__}")
