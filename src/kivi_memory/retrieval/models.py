"""Models for retrieval candidate merging and reranking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StructuredSignals:
    same_subject: bool = False
    shared_entity_count: int = 0
    same_predicate: bool = False
    same_memory_type: bool = False


@dataclass(frozen=True)
class StructuredMemoryCandidate:
    memory_id: str
    canonical_text: str
    structured_signals: StructuredSignals
    subject_entity_id: str | None = None
    predicate_type: str | None = None
    memory_type: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class GraphSignals:
    graph_distance: str | None = None
    graph_score: float = 0.0
    direct_seed_count: int = 0
    bridge_path_count: int = 0


@dataclass(frozen=True)
class GraphMemoryCandidate:
    memory_id: str
    canonical_text: str
    graph_signals: GraphSignals
    subject_entity_id: str | None = None
    predicate_type: str | None = None
    memory_type: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class MergedMemoryCandidate:
    memory_id: str
    canonical_text: str
    vector_similarity: float | None
    structured_signals: StructuredSignals
    graph_signals: GraphSignals = field(default_factory=GraphSignals)
    retrieval_sources: list[str] = field(default_factory=list)
    rerank_score: float = 0.0
    subject_entity_id: str | None = None
    predicate_type: str | None = None
    memory_type: str | None = None
    status: str | None = None
