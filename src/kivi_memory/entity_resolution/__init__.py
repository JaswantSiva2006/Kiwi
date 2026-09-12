"""Kivi entity-resolution public API."""

from kivi_memory.entity_resolution.candidates import (
    EntityCandidate,
    MatchType,
    find_entity_candidates,
)
from kivi_memory.entity_resolution.config import (
    ENTITY_AUTO_MATCH_THRESHOLD,
    ENTITY_FUZZY_CANDIDATE_THRESHOLD,
    ENTITY_MIN_SCORE_MARGIN,
    ENTITY_PLAUSIBLE_MATCH_THRESHOLD,
)
from kivi_memory.entity_resolution.normalizer import normalize_entity_name
from kivi_memory.entity_resolution.orchestrator import EntityOrCreateResult, resolve_or_create_entity
from kivi_memory.entity_resolution.repository import EntityAliasRecord, EntityNotFoundError, EntityRecord
from kivi_memory.entity_resolution.resolver import (
    EntityResolution,
    EntityResolutionResult,
    resolve_entity_mention,
)
from kivi_memory.entity_resolution.writes import add_entity_alias, create_entity

__all__ = [
    "ENTITY_AUTO_MATCH_THRESHOLD",
    "ENTITY_FUZZY_CANDIDATE_THRESHOLD",
    "ENTITY_MIN_SCORE_MARGIN",
    "ENTITY_PLAUSIBLE_MATCH_THRESHOLD",
    "EntityAliasRecord",
    "EntityCandidate",
    "EntityNotFoundError",
    "EntityOrCreateResult",
    "EntityRecord",
    "EntityResolution",
    "EntityResolutionResult",
    "MatchType",
    "add_entity_alias",
    "create_entity",
    "find_entity_candidates",
    "normalize_entity_name",
    "resolve_or_create_entity",
    "resolve_entity_mention",
]
