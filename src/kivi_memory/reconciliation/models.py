"""Typed results for memory reconciliation decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReconciliationOp(StrEnum):
    ADD = "ADD"
    REINFORCE = "REINFORCE"
    SUPERSEDE = "SUPERSEDE"
    RETRACT = "RETRACT"
    NO_MEMORY = "NO_MEMORY"


class ReconciliationFailure(RuntimeError):
    """Raised when an LLM reconciliation decision cannot be validated."""


@dataclass(frozen=True)
class ReconciliationDecision:
    op: ReconciliationOp
    target_memory_ids: list[str]
    reason: str
    candidate_map: dict[str, str] = field(default_factory=dict)
    compact_input: str | None = None
    candidate_hydration_ms: float = 0.0
    formatting_ms: float = 0.0
    llm_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "target_memory_ids": self.target_memory_ids,
            "reason": self.reason,
            "candidate_map": self.candidate_map,
            "compact_input": self.compact_input,
            "candidate_hydration_ms": self.candidate_hydration_ms,
            "formatting_ms": self.formatting_ms,
            "llm_ms": self.llm_ms,
            "validation_ms": self.validation_ms,
            "total_ms": self.total_ms,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
        }


@dataclass(frozen=True)
class HydratedMemoryArgument:
    role: str
    text: str
    is_entity: bool
    entity_id: str | None
    entity_type: str | None
    position: int


@dataclass(frozen=True)
class HydratedMemory:
    memory_id: str
    canonical_text: str
    subject_entity_id: str | None
    subject_text: str | None
    predicate_type: str
    memory_type: str
    modality: str
    polarity: str
    certainty: str
    temporal_kind: str | None
    valid_from: str | None
    valid_to: str | None
    event_time: str | None
    temporal_precision: str | None
    recurrence: str | None
    recurrence_specifics: str | None
    status: str
    arguments: list[HydratedMemoryArgument] = field(default_factory=list)


@dataclass(frozen=True)
class JudgeResult:
    raw_decision: dict[str, Any]
    llm_ms: float
    prompt_eval_count: int | None = None
    eval_count: int | None = None
