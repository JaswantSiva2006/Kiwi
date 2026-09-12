from __future__ import annotations

from kivi_memory.entity_resolution.candidates import (
    ENTITY_FUZZY_CANDIDATE_THRESHOLD,
    find_entity_candidates,
)
from kivi_memory.entity_resolution.repository import CandidateEvidence


class FakeRepository:
    def __init__(
        self,
        exact: list[CandidateEvidence] | None = None,
        fuzzy: list[CandidateEvidence] | None = None,
    ) -> None:
        self.exact = exact or []
        self.fuzzy = fuzzy or []
        self.fuzzy_calls: list[tuple[str, float, int]] = []

    def find_exact(self, normalized_mention: str, limit: int) -> list[CandidateEvidence]:
        return self.exact

    def find_fuzzy(
        self,
        normalized_mention: str,
        threshold: float,
        limit: int,
    ) -> list[CandidateEvidence]:
        self.fuzzy_calls.append((normalized_mention, threshold, limit))
        return self.fuzzy


def evidence(
    entity_id: str,
    canonical_name: str,
    entity_type: str,
    matched_text: str,
    match_type: str,
    similarity_score: float,
) -> CandidateEvidence:
    return CandidateEvidence(
        entity_id=entity_id,
        canonical_name=canonical_name,
        entity_type=entity_type,
        matched_text=matched_text,
        match_type=match_type,
        similarity_score=similarity_score,
    )


def test_unique_type_compatible_exact_candidate_is_sufficient() -> None:
    repo = FakeRepository(
        exact=[
            evidence("1", "Priya", "PERSON", "priya", "EXACT_NAME", 1.0),
            evidence("1", "Priya", "PERSON", "priya", "EXACT_ALIAS", 1.0),
        ],
        fuzzy=[evidence("2", "Priyam", "PERSON", "priyam", "FUZZY_NAME", 0.8)],
    )

    candidates = find_entity_candidates("Priya", "PERSON", repository=repo)

    assert len(candidates) == 1
    assert candidates[0].entity_id == "1"
    assert candidates[0].match_type == "EXACT_NAME"
    assert repo.fuzzy_calls == []


def test_type_mismatch_exact_candidate_triggers_fuzzy_but_is_retained() -> None:
    repo = FakeRepository(
        exact=[evidence("1", "Atlas", "PROJECT", "atlas", "EXACT_NAME", 1.0)],
        fuzzy=[evidence("2", "Atlas", "PERSON", "atlas", "FUZZY_NAME", 0.7)],
    )

    candidates = find_entity_candidates("Atlas", "PERSON", repository=repo)

    assert [candidate.entity_id for candidate in candidates] == ["1", "2"]
    assert candidates[0].type_match is False
    assert candidates[1].type_match is True
    assert repo.fuzzy_calls[0][1] == ENTITY_FUZZY_CANDIDATE_THRESHOLD


def test_fuzzy_aliases_are_collapsed_to_strongest_entity_evidence() -> None:
    repo = FakeRepository(
        fuzzy=[
            evidence("1", "Priya", "PERSON", "priya", "FUZZY_NAME", 0.7),
            evidence("1", "Priya", "PERSON", "priyaa", "FUZZY_ALIAS", 0.9),
            evidence("2", "Priyam", "PERSON", "priyam", "FUZZY_NAME", 0.8),
        ],
    )

    candidates = find_entity_candidates("Priyaa", "PERSON", repository=repo)

    assert [candidate.entity_id for candidate in candidates] == ["1", "2"]
    assert candidates[0].matched_text == "priyaa"
    assert candidates[0].match_type == "FUZZY_ALIAS"
