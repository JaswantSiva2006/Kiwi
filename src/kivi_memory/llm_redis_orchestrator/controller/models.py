"""Turn result models for the API/controller layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UserMessageResult:
    message_id: str
    text: str
    timestamp: str


@dataclass(frozen=True)
class AssistantMessageResult:
    message_id: str
    text: str
    timestamp: str


@dataclass(frozen=True)
class MemoryPreparationSummary:
    needs_long_term_memory: bool
    retrieval_queries: list[str]
    retrieved_memory_ids: list[str]


@dataclass(frozen=True)
class TurnDiagnostics:
    redis_user_append_ms: float
    redis_context_ms: float
    router_llm_ms: float
    memory_prepare_ms: float
    retrieval_ms: float
    final_agent_ms: float
    context_assembly_ms: float
    final_llm_ms: float
    sarvam_api_ms: float
    redis_assistant_append_ms: float
    total_turn_ms: float
    needs_long_term_memory: bool
    retrieval_query_count: int
    retrieved_memory_count: int


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    thread_id: str
    user_message: UserMessageResult
    memory_preparation: MemoryPreparationSummary
    assistant_message: AssistantMessageResult
    diagnostics: TurnDiagnostics
    metadata: dict[str, Any] = field(default_factory=dict)
