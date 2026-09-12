"""Models for the deterministic memory-entity graph projection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphSourceArgument:
    role: str
    is_entity: bool
    entity_id: str | None
    position: int


@dataclass(frozen=True)
class GraphSourceMemory:
    memory_id: str
    predicate_type: str
    subject_entity_id: str | None
    arguments: list[GraphSourceArgument]


@dataclass(frozen=True)
class MemoryEntityLink:
    memory_id: str
    entity_id: str
    link_type: str
    argument_role: str | None
    predicate_type: str
    position: int | None


@dataclass(frozen=True)
class GraphProjectionResult:
    memory_id: str
    links_considered: int
    links_created: int
    subject_links_created: int
    argument_links_created: int


@dataclass(frozen=True)
class GraphBackfillResult:
    total_active_memories: int
    projected_memories: int
    links_created: int
    failed: int
    failures: list[dict[str, str]] = field(default_factory=list)
