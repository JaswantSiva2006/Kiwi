"""Write-aware orchestration for entity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kivi_memory.entity_resolution.candidates import EntityCandidate
from kivi_memory.entity_resolution.repository import EntityRecord, EntityRepository
from kivi_memory.entity_resolution.resolver import (
    EntityResolution,
    EntityResolutionResult,
    resolve_entity_mention,
)


@dataclass(frozen=True)
class EntityOrCreateResult:
    mention: str
    normalized_mention: str
    resolution: EntityResolution
    entity_id: str | None
    candidates: list[EntityCandidate]
    reason: str
    created: bool


EntityResolver = Callable[[str, str | None], EntityResolutionResult]
EntityCreator = Callable[[str, str, str | None], EntityRecord]


def resolve_or_create_entity(
    mention: str,
    entity_type: str,
    source_episode_id: str | None = None,
    resolver: EntityResolver | None = None,
    creator: EntityCreator | None = None,
    repository: EntityRepository | None = None,
) -> EntityOrCreateResult:
    """Resolve a mention and create an entity only when resolution is NEW."""

    resolve = resolver or resolve_entity_mention
    resolution_result = resolve(mention, entity_type)

    if resolution_result.resolution == EntityResolution.MATCHED:
        return _from_resolution_result(resolution_result, created=False)

    if resolution_result.resolution == EntityResolution.AMBIGUOUS:
        return _from_resolution_result(resolution_result, created=False)

    create = creator
    if create is None:
        repo = repository or EntityRepository()
        create = repo.create_entity
    entity = create(mention, entity_type, source_episode_id)
    return EntityOrCreateResult(
        mention=resolution_result.mention,
        normalized_mention=resolution_result.normalized_mention,
        resolution=resolution_result.resolution,
        entity_id=entity.entity_id,
        candidates=resolution_result.candidates,
        reason=resolution_result.reason,
        created=True,
    )


def _from_resolution_result(
    result: EntityResolutionResult,
    created: bool,
) -> EntityOrCreateResult:
    return EntityOrCreateResult(
        mention=result.mention,
        normalized_mention=result.normalized_mention,
        resolution=result.resolution,
        entity_id=result.entity_id,
        candidates=result.candidates,
        reason=result.reason,
        created=created,
    )
