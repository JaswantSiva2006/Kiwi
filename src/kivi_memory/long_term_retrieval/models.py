"""Models for read-only long-term memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryEntityMatch:
    entity_id: str
    canonical_name: str
    entity_type: str
    matched_text: str
    match_method: str
    score: float


@dataclass(frozen=True)
class BranchCandidate:
    memory_id: str
    source: str
    rank: int
    raw_score: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedCandidate:
    memory_id: str
    rrf_score: float
    retrieval_sources: list[str]
    branch_ranks: dict[str, int | None]
    branch_scores: dict[str, float | None]
    best_branch_rank: int


@dataclass(frozen=True)
class HydratedRetrievedMemory:
    memory_id: str
    canonical_text: str
    status: str
    version: int
    rrf_score: float
    retrieval_sources: list[str]
    branch_ranks: dict[str, int | None]
    branch_scores: dict[str, float | None]
    subject: dict[str, Any]
    predicate_type: str
    memory_type: str
    modality: str | None
    polarity: str | None
    certainty: str | None
    explicitness: str | None
    attributed_to: str | None
    temporal: dict[str, Any]
    arguments: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    created_at: str | None = None


@dataclass(frozen=True)
class RetrievalDiagnostics:
    vector_ms: float = 0.0
    lexical_ms: float = 0.0
    entity_match_ms: float = 0.0
    structured_ms: float = 0.0
    graph_ms: float = 0.0
    fusion_ms: float = 0.0
    hydrate_ms: float = 0.0
    total_ms: float = 0.0
    vector_count: int = 0
    lexical_count: int = 0
    structured_count: int = 0
    graph_count: int = 0
    union_unique_count: int = 0
    hnsw_used: bool = False
    hnsw_fallback_used: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    resolved_query_entities: list[QueryEntityMatch]
    memories: list[HydratedRetrievedMemory]
    diagnostics: RetrievalDiagnostics
    branch_candidates: dict[str, list[BranchCandidate]] = field(default_factory=dict)
