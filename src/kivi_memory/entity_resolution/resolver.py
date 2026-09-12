"""Decision layer for entity-resolution candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from kivi_memory.entity_resolution.candidates import EntityCandidate, MatchType, find_entity_candidates
from kivi_memory.entity_resolution.config import (
    ENTITY_AUTO_MATCH_THRESHOLD,
    ENTITY_MIN_SCORE_MARGIN,
    ENTITY_PLAUSIBLE_MATCH_THRESHOLD,
)
from kivi_memory.entity_resolution.normalizer import normalize_entity_name
from kivi_memory.entity_resolution.repository import EntityRecord, EntityRepository

CURRENT_USER_SELF_REFERENCES = frozenset(
    {
        "user",
        "the user",
        "current user",
        "the current user",
        "this user",
        "i",
        "me",
        "myself",
    }
)


class EntityResolution(StrEnum):
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    NEW = "NEW"


@dataclass(frozen=True)
class EntityResolutionResult:
    mention: str
    normalized_mention: str
    resolution: EntityResolution
    entity_id: str | None
    candidates: list[EntityCandidate]
    reason: str


CandidateFinder = Callable[[str, str | None], list[EntityCandidate]]
CurrentUserLookup = Callable[[], EntityRecord | None]


def resolve_entity_mention(
    mention: str,
    entity_type: str | None = None,
    candidate_finder: CandidateFinder | None = None,
    current_user_lookup: CurrentUserLookup | None = None,
) -> EntityResolutionResult:
    """Resolve an entity mention to MATCHED, AMBIGUOUS, or NEW without writes."""

    normalized_mention = normalize_entity_name(mention)
    current_user = _resolve_current_user(normalized_mention, current_user_lookup)
    if current_user is not None:
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.MATCHED,
            entity_id=current_user.entity_id,
            candidates=[],
            reason="CURRENT_USER_SELF_REFERENCE",
        )

    finder = candidate_finder or find_entity_candidates
    candidates = finder(mention, entity_type)

    if not candidates:
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.NEW,
            entity_id=None,
            candidates=[],
            reason="NO_CANDIDATES",
        )

    exact_candidates = [candidate for candidate in candidates if _is_exact(candidate)]
    type_compatible_exact = [candidate for candidate in exact_candidates if candidate.type_match]
    if len(exact_candidates) == 1 and len(type_compatible_exact) == 1:
        selected = type_compatible_exact[0]
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.MATCHED,
            entity_id=selected.entity_id,
            candidates=candidates,
            reason="UNIQUE_EXACT_TYPE_MATCH",
        )

    if len(exact_candidates) > 1:
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.AMBIGUOUS,
            entity_id=None,
            candidates=candidates,
            reason="MULTIPLE_EXACT_CANDIDATES",
        )

    plausible_candidates = [
        candidate for candidate in candidates if candidate.similarity_score >= ENTITY_PLAUSIBLE_MATCH_THRESHOLD
    ]
    if not plausible_candidates:
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.NEW,
            entity_id=None,
            candidates=candidates,
            reason="NO_PLAUSIBLE_CANDIDATES",
        )

    top = plausible_candidates[0]
    if not _is_fuzzy(top):
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.AMBIGUOUS,
            entity_id=None,
            candidates=candidates,
            reason="UNRESOLVED_CANDIDATES",
        )

    if entity_type is not None and not top.type_match:
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.AMBIGUOUS,
            entity_id=None,
            candidates=candidates,
            reason="TOP_FUZZY_TYPE_MISMATCH",
        )

    if top.similarity_score < ENTITY_AUTO_MATCH_THRESHOLD:
        return EntityResolutionResult(
            mention=mention,
            normalized_mention=normalized_mention,
            resolution=EntityResolution.AMBIGUOUS,
            entity_id=None,
            candidates=candidates,
            reason="TOP_FUZZY_BELOW_AUTO_MATCH_THRESHOLD",
        )

    if len(plausible_candidates) > 1:
        margin = top.similarity_score - plausible_candidates[1].similarity_score
        if margin < ENTITY_MIN_SCORE_MARGIN:
            return EntityResolutionResult(
                mention=mention,
                normalized_mention=normalized_mention,
                resolution=EntityResolution.AMBIGUOUS,
                entity_id=None,
                candidates=candidates,
                reason="FUZZY_SCORE_MARGIN_TOO_SMALL",
            )

    return EntityResolutionResult(
        mention=mention,
        normalized_mention=normalized_mention,
        resolution=EntityResolution.MATCHED,
        entity_id=top.entity_id,
        candidates=candidates,
        reason="STRONG_ISOLATED_FUZZY_MATCH",
    )


def _is_exact(candidate: EntityCandidate) -> bool:
    return candidate.match_type in {MatchType.EXACT_NAME, MatchType.EXACT_ALIAS}


def _is_fuzzy(candidate: EntityCandidate) -> bool:
    return candidate.match_type in {MatchType.FUZZY_NAME, MatchType.FUZZY_ALIAS}


def _resolve_current_user(
    normalized_mention: str,
    current_user_lookup: CurrentUserLookup | None,
) -> EntityRecord | None:
    if normalized_mention not in CURRENT_USER_SELF_REFERENCES:
        return None

    lookup = current_user_lookup
    if lookup is None:
        lookup = EntityRepository().get_current_user_entity
    return lookup()
