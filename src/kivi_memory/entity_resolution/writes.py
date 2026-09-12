"""Public write-side primitives for entity resolution."""

from __future__ import annotations

from kivi_memory.entity_resolution.repository import EntityAliasRecord, EntityRecord, EntityRepository


def create_entity(
    canonical_name: str,
    entity_type: str,
    source_episode_id: str | None = None,
    repository: EntityRepository | None = None,
) -> EntityRecord:
    repo = repository or EntityRepository()
    return repo.create_entity(canonical_name, entity_type, source_episode_id)


def add_entity_alias(
    entity_id: str,
    alias: str,
    confidence: float | None = None,
    alias_source: str | None = None,
    source_episode_id: str | None = None,
    repository: EntityRepository | None = None,
) -> EntityAliasRecord:
    repo = repository or EntityRepository()
    return repo.add_entity_alias(entity_id, alias, confidence, alias_source, source_episode_id)
