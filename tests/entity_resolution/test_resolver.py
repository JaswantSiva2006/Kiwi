from __future__ import annotations

from kivi_memory.entity_resolution import EntityResolution, resolve_entity_mention
from kivi_memory.entity_resolution.candidates import EntityCandidate, MatchType
from kivi_memory.entity_resolution.repository import EntityRecord


def candidate(
    entity_id: str,
    match_type: MatchType,
    similarity_score: float,
    type_match: bool = True,
) -> EntityCandidate:
    return EntityCandidate(
        entity_id=entity_id,
        canonical_name=f"Entity {entity_id}",
        entity_type="PERSON" if type_match else "PROJECT",
        matched_text=f"entity {entity_id}",
        match_type=match_type,
        similarity_score=similarity_score,
        type_match=type_match,
    )


def finder(candidates: list[EntityCandidate]):
    def _find(mention: str, entity_type: str | None) -> list[EntityCandidate]:
        return candidates

    return _find


def current_user() -> EntityRecord:
    return EntityRecord(
        entity_id="current-user-1",
        canonical_name="user",
        normalized_name="user",
        entity_type="person",
    )


def test_unique_exact_match_is_matched() -> None:
    result = resolve_entity_mention(
        "Priya",
        "PERSON",
        candidate_finder=finder([candidate("1", MatchType.EXACT_ALIAS, 1.0)]),
    )

    assert result.resolution == EntityResolution.MATCHED
    assert result.entity_id == "1"
    assert result.reason == "UNIQUE_EXACT_TYPE_MATCH"


def test_two_entities_sharing_same_exact_alias_are_ambiguous() -> None:
    result = resolve_entity_mention(
        "Priya",
        "PERSON",
        candidate_finder=finder(
            [
                candidate("1", MatchType.EXACT_ALIAS, 1.0),
                candidate("2", MatchType.EXACT_ALIAS, 1.0),
            ]
        ),
    )

    assert result.resolution == EntityResolution.AMBIGUOUS
    assert result.entity_id is None
    assert result.reason == "MULTIPLE_EXACT_CANDIDATES"


def test_no_candidates_is_new() -> None:
    result = resolve_entity_mention("New Person", "PERSON", candidate_finder=finder([]))

    assert result.resolution == EntityResolution.NEW
    assert result.entity_id is None
    assert result.candidates == []
    assert result.reason == "NO_CANDIDATES"


def test_strong_isolated_fuzzy_match_is_matched() -> None:
    result = resolve_entity_mention(
        "Priyaa",
        "PERSON",
        candidate_finder=finder([candidate("1", MatchType.FUZZY_ALIAS, 0.91)]),
    )

    assert result.resolution == EntityResolution.MATCHED
    assert result.entity_id == "1"
    assert result.reason == "STRONG_ISOLATED_FUZZY_MATCH"


def test_fuzzy_score_below_plausible_threshold_is_new() -> None:
    result = resolve_entity_mention(
        "Riya",
        "PERSON",
        candidate_finder=finder([candidate("1", MatchType.FUZZY_ALIAS, 0.375)]),
    )

    assert result.resolution == EntityResolution.NEW
    assert result.entity_id is None
    assert result.reason == "NO_PLAUSIBLE_CANDIDATES"


def test_fuzzy_score_between_plausible_and_auto_match_threshold_is_ambiguous() -> None:
    result = resolve_entity_mention(
        "Priyaa",
        "PERSON",
        candidate_finder=finder([candidate("1", MatchType.FUZZY_ALIAS, 0.75)]),
    )

    assert result.resolution == EntityResolution.AMBIGUOUS
    assert result.entity_id is None
    assert result.reason == "TOP_FUZZY_BELOW_AUTO_MATCH_THRESHOLD"


def test_top_two_fuzzy_candidates_within_score_margin_are_ambiguous() -> None:
    result = resolve_entity_mention(
        "Priyaa",
        "PERSON",
        candidate_finder=finder(
            [
                candidate("1", MatchType.FUZZY_ALIAS, 0.91),
                candidate("2", MatchType.FUZZY_ALIAS, 0.83),
            ]
        ),
    )

    assert result.resolution == EntityResolution.AMBIGUOUS
    assert result.entity_id is None
    assert result.reason == "FUZZY_SCORE_MARGIN_TOO_SMALL"


def test_type_mismatch_prevents_fuzzy_auto_match() -> None:
    result = resolve_entity_mention(
        "Atlas",
        "PERSON",
        candidate_finder=finder([candidate("1", MatchType.FUZZY_NAME, 0.95, type_match=False)]),
    )

    assert result.resolution == EntityResolution.AMBIGUOUS
    assert result.entity_id is None
    assert result.reason == "TOP_FUZZY_TYPE_MISMATCH"


def test_user_and_the_user_resolve_to_same_stable_entity_without_candidate_lookup() -> None:
    calls: list[str] = []

    def should_not_run(mention: str, entity_type: str | None) -> list[EntityCandidate]:
        calls.append(mention)
        return []

    user_result = resolve_entity_mention(
        "user",
        "PERSON",
        candidate_finder=should_not_run,
        current_user_lookup=current_user,
    )
    the_user_result = resolve_entity_mention(
        "the user",
        "PERSON",
        candidate_finder=should_not_run,
        current_user_lookup=current_user,
    )

    assert user_result.resolution == EntityResolution.MATCHED
    assert the_user_result.resolution == EntityResolution.MATCHED
    assert user_result.entity_id == the_user_result.entity_id == "current-user-1"
    assert user_result.reason == the_user_result.reason == "CURRENT_USER_SELF_REFERENCE"
    assert calls == []


def test_result_includes_normalized_mention() -> None:
    result = resolve_entity_mention(
        "  Priyaa!! ",
        "PERSON",
        candidate_finder=finder([candidate("1", MatchType.FUZZY_ALIAS, 0.91)]),
    )

    assert result.normalized_mention == "priyaa"
