"""Bounded graph retrieval over the memory_entity_links projection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from kivi_memory.common.schemas import CandidateSemanticAssertion
from kivi_memory.entity_resolution.repository import connect
from kivi_memory.entity_resolution.resolver import CURRENT_USER_SELF_REFERENCES
from kivi_memory.retrieval.config import MAX_GRAPH_TOP_K, get_graph_max_bridge_degree, get_graph_top_k
from kivi_memory.retrieval.models import GraphMemoryCandidate, GraphSignals

GRAPH_DISTANCE_DIRECT = "DIRECT"
GRAPH_DISTANCE_ONE_EXPANSION = "ONE_EXPANSION"
SOURCE_GRAPH = "GRAPH"
CURRENT_USER_REASON = "CURRENT_USER_SELF_REFERENCE"


@dataclass(frozen=True)
class GraphRetrievalResult:
    candidates: list[GraphMemoryCandidate]
    graph_retrieval_ms: float
    graph_seed_count: int
    graph_direct_count: int
    graph_expanded_count: int
    graph_returned_count: int


class GraphMemoryRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def retrieve_candidates(
        self,
        *,
        seed_entity_ids: list[str],
        max_bridge_degree: int,
        top_k: int,
    ) -> list[GraphMemoryCandidate]:
        """Return DIRECT and one-expansion graph candidates in one bounded query."""

        sql = """
            WITH seed_entities AS (
                SELECT DISTINCT unnest(%s::uuid[]) AS entity_id
            ),
            current_user_entity AS (
                (
                    SELECT e.entity_id
                    FROM entities e
                    WHERE e.normalized_name = 'user'
                    ORDER BY e.canonical_name
                    LIMIT 1
                )
                UNION ALL
                (
                    SELECT e.entity_id
                    FROM entity_aliases a
                    JOIN entities e ON e.entity_id = a.entity_id
                    WHERE a.normalized_alias = 'user'
                    ORDER BY e.canonical_name
                    LIMIT 1
                )
                LIMIT 1
            ),
            direct AS (
                SELECT
                    sm.memory_id,
                    sm.canonical_text,
                    sm.subject_entity_id,
                    sm.predicate_type,
                    sm.memory_type,
                    sm.status,
                    sm.updated_at,
                    COUNT(DISTINCT mel.entity_id)::int AS direct_seed_count
                FROM memory_entity_links mel
                JOIN seed_entities se ON se.entity_id = mel.entity_id
                JOIN semantic_memories sm ON sm.memory_id = mel.memory_id
                WHERE sm.status = 'ACTIVE'
                GROUP BY
                    sm.memory_id,
                    sm.canonical_text,
                    sm.subject_entity_id,
                    sm.predicate_type,
                    sm.memory_type,
                    sm.status,
                    sm.updated_at
            ),
            bridge_entities AS (
                SELECT DISTINCT
                    related.entity_id AS bridge_entity_id,
                    direct.memory_id AS via_memory_id
                FROM direct
                JOIN memory_entity_links related ON related.memory_id = direct.memory_id
                LEFT JOIN current_user_entity cu ON cu.entity_id = related.entity_id
                WHERE related.entity_id <> ALL(%s::uuid[])
                  AND cu.entity_id IS NULL
            ),
            bounded_bridge_entities AS (
                SELECT
                    be.bridge_entity_id,
                    be.via_memory_id
                FROM bridge_entities be
                JOIN memory_entity_links degree_link ON degree_link.entity_id = be.bridge_entity_id
                JOIN semantic_memories degree_memory
                  ON degree_memory.memory_id = degree_link.memory_id
                 AND degree_memory.status = 'ACTIVE'
                GROUP BY be.bridge_entity_id, be.via_memory_id
                HAVING COUNT(DISTINCT degree_link.memory_id) <= %s
            ),
            expanded AS (
                SELECT
                    sm.memory_id,
                    sm.canonical_text,
                    sm.subject_entity_id,
                    sm.predicate_type,
                    sm.memory_type,
                    sm.status,
                    sm.updated_at,
                    COUNT(DISTINCT (bbe.bridge_entity_id::text || ':' || bbe.via_memory_id::text))::int AS bridge_path_count
                FROM bounded_bridge_entities bbe
                JOIN memory_entity_links target_link ON target_link.entity_id = bbe.bridge_entity_id
                JOIN semantic_memories sm
                  ON sm.memory_id = target_link.memory_id
                 AND sm.status = 'ACTIVE'
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM direct d
                    WHERE d.memory_id = sm.memory_id
                )
                GROUP BY
                    sm.memory_id,
                    sm.canonical_text,
                    sm.subject_entity_id,
                    sm.predicate_type,
                    sm.memory_type,
                    sm.status,
                    sm.updated_at
            ),
            unioned AS (
                SELECT
                    memory_id,
                    canonical_text,
                    subject_entity_id,
                    predicate_type,
                    memory_type,
                    status,
                    updated_at,
                    'DIRECT'::text AS graph_distance,
                    1.0::double precision AS graph_score,
                    direct_seed_count,
                    0::int AS bridge_path_count
                FROM direct
                UNION ALL
                SELECT
                    memory_id,
                    canonical_text,
                    subject_entity_id,
                    predicate_type,
                    memory_type,
                    status,
                    updated_at,
                    'ONE_EXPANSION'::text AS graph_distance,
                    0.5::double precision AS graph_score,
                    0::int AS direct_seed_count,
                    bridge_path_count
                FROM expanded
            )
            SELECT
                memory_id::text AS memory_id,
                canonical_text,
                subject_entity_id::text AS subject_entity_id,
                predicate_type,
                memory_type,
                status,
                graph_distance,
                graph_score,
                direct_seed_count,
                bridge_path_count
            FROM unioned
            ORDER BY
                graph_score DESC,
                direct_seed_count DESC,
                bridge_path_count DESC,
                updated_at DESC,
                memory_id DESC
            LIMIT %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (seed_entity_ids, seed_entity_ids, max_bridge_degree, top_k))
                rows = cur.fetchall()
        return [_row_to_graph_candidate(row) for row in rows]


def retrieve_graph_candidates(
    assertion: CandidateSemanticAssertion | dict[str, Any],
    entity_resolution: dict[str, Any],
    top_k: int | None = None,
    max_bridge_degree: int | None = None,
    repository: GraphMemoryRepository | None = None,
) -> GraphRetrievalResult:
    """Retrieve active memories connected to resolved entities by one graph hop."""

    if top_k is None:
        top_k = get_graph_top_k()
    if top_k < 1 or top_k > MAX_GRAPH_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_GRAPH_TOP_K}")
    if max_bridge_degree is None:
        max_bridge_degree = get_graph_max_bridge_degree()
    if max_bridge_degree < 1:
        raise ValueError("max_bridge_degree must be positive")

    assertion = CandidateSemanticAssertion.model_validate(assertion)
    seeds = _resolved_seed_entities(assertion, entity_resolution)
    started = time.perf_counter()
    if not seeds:
        return GraphRetrievalResult([], 0.0, 0, 0, 0, 0)

    repo = repository or GraphMemoryRepository()
    candidates = repo.retrieve_candidates(
        seed_entity_ids=seeds,
        max_bridge_degree=max_bridge_degree,
        top_k=top_k,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return GraphRetrievalResult(
        candidates=candidates,
        graph_retrieval_ms=elapsed_ms,
        graph_seed_count=len(seeds),
        graph_direct_count=sum(1 for candidate in candidates if candidate.graph_signals.graph_distance == GRAPH_DISTANCE_DIRECT),
        graph_expanded_count=sum(
            1 for candidate in candidates if candidate.graph_signals.graph_distance == GRAPH_DISTANCE_ONE_EXPANSION
        ),
        graph_returned_count=len(candidates),
    )


def _resolved_seed_entities(
    assertion: CandidateSemanticAssertion,
    entity_resolution: dict[str, Any],
) -> list[str]:
    del assertion
    seed_entries: list[tuple[str, bool]] = []
    subject = entity_resolution.get("subject") or {}
    _append_seed(seed_entries, subject)
    for argument in entity_resolution.get("semantic_arguments") or []:
        _append_seed(seed_entries, argument.get("result") or {})

    has_non_self = any(not is_self for _, is_self in seed_entries)
    result = []
    seen = set()
    for entity_id, is_self in seed_entries:
        if has_non_self and is_self:
            continue
        if entity_id not in seen:
            seen.add(entity_id)
            result.append(entity_id)
    return result


def _append_seed(seed_entries: list[tuple[str, bool]], result: dict[str, Any]) -> None:
    entity_id = _get(result, "entity_id")
    if _enum_value(_get(result, "resolution")) not in {"MATCHED", "NEW"} or not entity_id:
        return
    reason = _get(result, "reason")
    normalized = _get(result, "normalized_mention")
    is_self = reason == CURRENT_USER_REASON or normalized in CURRENT_USER_SELF_REFERENCES
    seed_entries.append((str(entity_id), is_self))


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _row_to_graph_candidate(row: dict[str, Any]) -> GraphMemoryCandidate:
    return GraphMemoryCandidate(
        memory_id=row["memory_id"],
        canonical_text=row["canonical_text"],
        graph_signals=GraphSignals(
            graph_distance=row["graph_distance"],
            graph_score=float(row["graph_score"]),
            direct_seed_count=int(row["direct_seed_count"] or 0),
            bridge_path_count=int(row["bridge_path_count"] or 0),
        ),
        subject_entity_id=row["subject_entity_id"],
        predicate_type=row["predicate_type"],
        memory_type=row["memory_type"],
        status=row["status"],
    )
