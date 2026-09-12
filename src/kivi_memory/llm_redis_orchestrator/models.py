"""Return models for current-thread to long-term-memory preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouterDecision:
    needs_long_term_memory: bool
    retrieval_queries: list[str]


@dataclass(frozen=True)
class MergedRetrievedMemory:
    memory_id: str
    canonical_text: str
    subject: dict[str, Any]
    arguments: list[dict[str, Any]]
    temporal: dict[str, Any]
    evidence: list[dict[str, Any]]
    matched_retrieval_queries: list[str]
    predicate_type: str | None = None
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPreparationDiagnostics:
    redis_context_ms: float
    router_llm_ms: float
    retrieval_ms: float
    total_ms: float
    router_model_call_count: int
    retrieval_call_count: int
    redis_context_message_count: int
    router_fallback_used: bool = False
    router_error: str | None = None


@dataclass(frozen=True)
class MemoryPreparationResult:
    thread_id: str
    current_query: str
    needs_long_term_memory: bool
    retrieval_queries: list[str]
    retrieved_memories: list[MergedRetrievedMemory]
    diagnostics: MemoryPreparationDiagnostics
