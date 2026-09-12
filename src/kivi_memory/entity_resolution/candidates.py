"""Entity candidate generation and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kivi_memory.entity_resolution.config import ENTITY_FUZZY_CANDIDATE_THRESHOLD
from kivi_memory.entity_resolution.normalizer import normalize_entity_name
from kivi_memory.entity_resolution.repository import CandidateEvidence, EntityRepository


class MatchType(StrEnum):
    EXACT_NAME = "EXACT_NAME"
    EXACT_ALIAS = "EXACT_ALIAS"
    FUZZY_NAME = "FUZZY_NAME"
    FUZZY_ALIAS = "FUZZY_ALIAS"


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    canonical_name: str
    entity_type: str
    matched_text: str
    match_type: MatchType
    similarity_score: float
    type_match: bool


def find_entity_candidates(
    mention: str,
    entity_type: str | None = None,
    limit: int = 10,
    repository: EntityRepository | None = None,
) -> list[EntityCandidate]:
    """Find ranked entity candidates without making a final resolution decision."""

    if limit < 1:
        raise ValueError("limit must be at least 1")

    normalized_mention = normalize_entity_name(mention)
    if not normalized_mention:
        return []

    repo = repository or EntityRepository()
    fetch_limit = max(limit * 4, 20)
    exact_evidence = repo.find_exact(normalized_mention, fetch_limit)
    exact_candidates = _collapse_evidence(exact_evidence, entity_type)
    type_compatible_exact = [candidate for candidate in exact_candidates if candidate.type_match]
    exact_is_sufficient = len(exact_candidates) == 1 and len(type_compatible_exact) == 1

    if exact_is_sufficient:
        return exact_candidates[:limit]

    fuzzy_evidence = repo.find_fuzzy(
        normalized_mention,
        ENTITY_FUZZY_CANDIDATE_THRESHOLD,
        fetch_limit,
    )
    candidates = _collapse_evidence([*exact_evidence, *fuzzy_evidence], entity_type)
    return candidates[:limit]


def _collapse_evidence(
    evidence_rows: list[CandidateEvidence],
    requested_entity_type: str | None,
) -> list[EntityCandidate]:
    candidates_by_entity_id: dict[str, EntityCandidate] = {}
    for evidence in evidence_rows:
        candidate = EntityCandidate(
            entity_id=evidence.entity_id,
            canonical_name=evidence.canonical_name,
            entity_type=evidence.entity_type,
            matched_text=evidence.matched_text,
            match_type=MatchType(evidence.match_type),
            similarity_score=evidence.similarity_score,
            type_match=_type_matches(requested_entity_type, evidence.entity_type),
        )
        current = candidates_by_entity_id.get(candidate.entity_id)
        if current is None or _candidate_sort_key(candidate) > _candidate_sort_key(current):
            candidates_by_entity_id[candidate.entity_id] = candidate

    return sorted(
        candidates_by_entity_id.values(),
        key=_candidate_sort_key,
        reverse=True,
    )


def _type_matches(requested_entity_type: str | None, stored_entity_type: str) -> bool:
    if requested_entity_type is None:
        return True
    return requested_entity_type.casefold() == stored_entity_type.casefold()


def _candidate_sort_key(candidate: EntityCandidate) -> tuple[int, int, float, int, str]:
    return (
        int(candidate.match_type in {MatchType.EXACT_NAME, MatchType.EXACT_ALIAS}),
        int(candidate.type_match),
        candidate.similarity_score,
        _match_type_rank(candidate.match_type),
        candidate.canonical_name.casefold(),
    )


def _match_type_rank(match_type: MatchType) -> int:
    return {
        MatchType.EXACT_NAME: 4,
        MatchType.EXACT_ALIAS: 3,
        MatchType.FUZZY_NAME: 2,
        MatchType.FUZZY_ALIAS: 1,
    }[match_type]
