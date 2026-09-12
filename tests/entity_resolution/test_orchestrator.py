from __future__ import annotations

from kivi_memory.entity_resolution import EntityResolution, resolve_or_create_entity
from kivi_memory.entity_resolution.candidates import EntityCandidate, MatchType
from kivi_memory.entity_resolution.normalizer import normalize_entity_name
from kivi_memory.entity_resolution.repository import EntityRecord
from kivi_memory.entity_resolution.resolver import EntityResolutionResult


def candidate(entity_id: str, match_type: MatchType = MatchType.EXACT_NAME) -> EntityCandidate:
    return EntityCandidate(
        entity_id=entity_id,
        canonical_name="Priya Nair",
        entity_type="PERSON",
        matched_text="priya nair",
        match_type=match_type,
        similarity_score=1.0,
        type_match=True,
    )


def resolution_result(
    mention: str,
    resolution: EntityResolution,
    entity_id: str | None = None,
    candidates: list[EntityCandidate] | None = None,
    reason: str = "TEST_REASON",
) -> EntityResolutionResult:
    return EntityResolutionResult(
        mention=mention,
        normalized_mention=normalize_entity_name(mention),
        resolution=resolution,
        entity_id=entity_id,
        candidates=candidates or [],
        reason=reason,
    )


def test_matched_returns_existing_entity_without_creating() -> None:
    created: list[tuple[str, str, str | None]] = []

    def resolver(mention: str, entity_type: str | None) -> EntityResolutionResult:
        return resolution_result(
            mention,
            EntityResolution.MATCHED,
            entity_id="existing-1",
            candidates=[candidate("existing-1")],
            reason="UNIQUE_EXACT_TYPE_MATCH",
        )

    def creator(canonical_name: str, entity_type: str, source_episode_id: str | None) -> EntityRecord:
        created.append((canonical_name, entity_type, source_episode_id))
        return EntityRecord("new-1", canonical_name, normalize_entity_name(canonical_name), entity_type)

    result = resolve_or_create_entity("Priya Nair", "PERSON", resolver=resolver, creator=creator)

    assert result.resolution == EntityResolution.MATCHED
    assert result.entity_id == "existing-1"
    assert result.created is False
    assert created == []


def test_ambiguous_returns_candidates_without_creating() -> None:
    created: list[tuple[str, str, str | None]] = []
    candidates = [candidate("1"), candidate("2")]

    def resolver(mention: str, entity_type: str | None) -> EntityResolutionResult:
        return resolution_result(
            mention,
            EntityResolution.AMBIGUOUS,
            candidates=candidates,
            reason="MULTIPLE_EXACT_CANDIDATES",
        )

    def creator(canonical_name: str, entity_type: str, source_episode_id: str | None) -> EntityRecord:
        created.append((canonical_name, entity_type, source_episode_id))
        return EntityRecord("new-1", canonical_name, normalize_entity_name(canonical_name), entity_type)

    result = resolve_or_create_entity("Priya", "PERSON", resolver=resolver, creator=creator)

    assert result.resolution == EntityResolution.AMBIGUOUS
    assert result.entity_id is None
    assert result.candidates == candidates
    assert result.created is False
    assert created == []


def test_new_creates_entity_with_original_mention_as_canonical_name() -> None:
    created: list[tuple[str, str, str | None]] = []

    def resolver(mention: str, entity_type: str | None) -> EntityResolutionResult:
        return resolution_result(mention, EntityResolution.NEW, reason="NO_CANDIDATES")

    def creator(canonical_name: str, entity_type: str, source_episode_id: str | None) -> EntityRecord:
        created.append((canonical_name, entity_type, source_episode_id))
        return EntityRecord("created-1", canonical_name, normalize_entity_name(canonical_name), entity_type)

    result = resolve_or_create_entity(
        "Priya Nair",
        "PERSON",
        source_episode_id="ep-1",
        resolver=resolver,
        creator=creator,
    )

    assert result.resolution == EntityResolution.NEW
    assert result.entity_id == "created-1"
    assert result.created is True
    assert created == [("Priya Nair", "PERSON", "ep-1")]


def test_repeated_call_after_new_resolves_created_entity_without_duplicate() -> None:
    created_entities: dict[str, EntityRecord] = {}

    def resolver(mention: str, entity_type: str | None) -> EntityResolutionResult:
        normalized = normalize_entity_name(mention)
        entity = created_entities.get(normalized)
        if entity is None:
            return resolution_result(mention, EntityResolution.NEW, reason="NO_CANDIDATES")
        return resolution_result(
            mention,
            EntityResolution.MATCHED,
            entity_id=entity.entity_id,
            candidates=[candidate(entity.entity_id)],
            reason="UNIQUE_EXACT_TYPE_MATCH",
        )

    def creator(canonical_name: str, entity_type: str, source_episode_id: str | None) -> EntityRecord:
        entity = EntityRecord(
            entity_id=f"created-{len(created_entities) + 1}",
            canonical_name=canonical_name,
            normalized_name=normalize_entity_name(canonical_name),
            entity_type=entity_type,
        )
        created_entities[entity.normalized_name] = entity
        return entity

    first = resolve_or_create_entity("Priya Nair", "PERSON", resolver=resolver, creator=creator)
    second = resolve_or_create_entity("Priya Nair", "PERSON", resolver=resolver, creator=creator)

    assert first.resolution == EntityResolution.NEW
    assert first.entity_id == "created-1"
    assert first.created is True
    assert second.resolution == EntityResolution.MATCHED
    assert second.entity_id == "created-1"
    assert second.created is False
    assert len(created_entities) == 1
