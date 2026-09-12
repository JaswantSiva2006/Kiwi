"""Protocol boundary for the future final Hey Kivi response agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from kivi_memory.llm_redis_orchestrator.models import MemoryPreparationResult
from kivi_memory.working_memory.models import ThreadMessage


@dataclass(frozen=True)
class AgentResponse:
    text: str
    message_id: str | None = None
    used_long_term_memory: bool = False
    supplied_memory_ids: list[str] = field(default_factory=list)
    supplied_evidence_message_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class HeyKiviAgent(Protocol):
    def respond(
        self,
        *,
        thread_id: str,
        current_query: str,
        prior_thread_messages: list[ThreadMessage],
        memory_context: MemoryPreparationResult,
    ) -> AgentResponse:
        """Return the user-facing assistant response for one turn."""
