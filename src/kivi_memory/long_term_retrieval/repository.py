"""Read-only PostgreSQL queries for long-term memory retrieval."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from kivi_memory.entity_resolution.normalizer import normalize_entity_name
from kivi_memory.entity_resolution.repository import connect
from kivi_memory.long_term_retrieval.models import BranchCandidate, HydratedRetrievedMemory, QueryEntityMatch
from kivi_memory.retrieval.config import get_graph_max_bridge_degree
from kivi_memory.retrieval.graph import GraphMemoryRepository

SOURCE_LEXICAL = "LEXICAL"
SOURCE_STRUCTURED = "STRUCTURED"
SOURCE_GRAPH = "GRAPH"


class LongTermRetrievalRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def connect(self):
        return connect(self.database_url)

    def match_query_entities(self, query_text: str) -> list[QueryEntityMatch]:
        normalized_query = normalize_entity_name(query_text)
        if not normalized_query:
            return []

        rows = self._query_entity_match_rows(normalized_query)
        exact = _dedupe_entity_matches([row for row in rows if row["match_method"] in {"exact_name", "exact_alias"}])
        if exact:
            return exact
        return _dedupe_entity_matches([row for row in rows if row["match_method"] == "fuzzy"])

    def _query_entity_match_rows(self, normalized_query: str) -> list[dict[str, Any]]:
        sql = """
            WITH query AS (
                SELECT %s::text AS normalized_query
            ),
            evidence AS (
                SELECT
                    e.entity_id,
                    e.canonical_name,
                    e.entity_type,
                    e.normalized_name AS matched_text,
                    'exact_name'::text AS match_method,
                    1.0::double precision AS score
                FROM entities e, query q
                WHERE (' ' || q.normalized_query || ' ') LIKE ('%% ' || e.normalized_name || ' %%')
                UNION ALL
                SELECT
                    e.entity_id,
                    e.canonical_name,
                    e.entity_type,
                    a.normalized_alias AS matched_text,
                    'exact_alias'::text AS match_method,
                    1.0::double precision AS score
                FROM entity_aliases a
                JOIN entities e ON e.entity_id = a.entity_id, query q
                WHERE (' ' || q.normalized_query || ' ') LIKE ('%% ' || a.normalized_alias || ' %%')
                UNION ALL
                SELECT
                    e.entity_id,
                    e.canonical_name,
                    e.entity_type,
                    e.normalized_name AS matched_text,
                    'fuzzy'::text AS match_method,
                    word_similarity(e.normalized_name, q.normalized_query) AS score
                FROM entities e, query q
                WHERE word_similarity(e.normalized_name, q.normalized_query) >= %s
                UNION ALL
                SELECT
                    e.entity_id,
                    e.canonical_name,
                    e.entity_type,
                    a.normalized_alias AS matched_text,
                    'fuzzy'::text AS match_method,
                    word_similarity(a.normalized_alias, q.normalized_query) AS score
                FROM entity_aliases a
                JOIN entities e ON e.entity_id = a.entity_id, query q
                WHERE word_similarity(a.normalized_alias, q.normalized_query) >= %s
            )
            SELECT
                entity_id::text AS entity_id,
                canonical_name,
                entity_type,
                matched_text,
                match_method,
                max(score) AS score
            FROM evidence
            GROUP BY entity_id, canonical_name, entity_type, matched_text, match_method
            ORDER BY
                CASE match_method WHEN 'exact_name' THEN 0 WHEN 'exact_alias' THEN 1 ELSE 2 END,
                score DESC,
                length(matched_text) DESC,
                canonical_name
        """
        from kivi_memory.long_term_retrieval.config import QUERY_ENTITY_FUZZY_THRESHOLD

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (normalized_query, QUERY_ENTITY_FUZZY_THRESHOLD, QUERY_ENTITY_FUZZY_THRESHOLD))
                return [dict(row) for row in cur.fetchall()]

    def retrieve_lexical_candidates(self, query_text: str, branch_k: int) -> list[BranchCandidate]:
        sql = """
            WITH query AS (
                SELECT
                    plainto_tsquery('simple', %s) AS tsq,
                    %s::text AS raw_query
            ),
            ranked AS (
                SELECT
                    sm.memory_id::text AS memory_id,
                    ts_rank_cd(to_tsvector('simple', sm.canonical_text), q.tsq) AS fts_rank,
                    similarity(sm.canonical_text, q.raw_query) AS trigram_similarity
                FROM semantic_memories sm, query q
                WHERE sm.status = 'ACTIVE'
                  AND (
                    to_tsvector('simple', sm.canonical_text) @@ q.tsq
                    OR similarity(sm.canonical_text, q.raw_query) > 0
                  )
            )
            SELECT
                memory_id,
                (fts_rank + trigram_similarity) AS lexical_score,
                fts_rank,
                trigram_similarity
            FROM ranked
            ORDER BY lexical_score DESC, fts_rank DESC, trigram_similarity DESC, memory_id ASC
            LIMIT %s
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query_text, query_text, branch_k))
                rows = cur.fetchall()
        return [
            BranchCandidate(
                memory_id=row["memory_id"],
                source=SOURCE_LEXICAL,
                rank=index,
                raw_score=float(row["lexical_score"]),
                diagnostics={
                    "fts_rank": float(row["fts_rank"]),
                    "trigram_similarity": float(row["trigram_similarity"]),
                },
            )
            for index, row in enumerate(rows, start=1)
        ]

    def retrieve_structured_candidates(self, entity_ids: list[str], branch_k: int) -> list[BranchCandidate]:
        if not entity_ids:
            return []
        sql = """
            WITH query_entities AS (
                SELECT DISTINCT unnest(%s::uuid[]) AS entity_id
            ),
            subject_matches AS (
                SELECT sm.memory_id, COUNT(DISTINCT sm.subject_entity_id)::int AS subject_match_count
                FROM semantic_memories sm
                JOIN query_entities qe ON qe.entity_id = sm.subject_entity_id
                WHERE sm.status = 'ACTIVE'
                GROUP BY sm.memory_id
            ),
            argument_matches AS (
                SELECT ma.memory_id, COUNT(DISTINCT ma.entity_id)::int AS argument_match_count
                FROM memory_arguments ma
                JOIN semantic_memories sm ON sm.memory_id = ma.memory_id
                JOIN query_entities qe ON qe.entity_id = ma.entity_id
                WHERE sm.status = 'ACTIVE'
                  AND ma.is_entity = true
                GROUP BY ma.memory_id
            ),
            merged AS (
                SELECT
                    COALESCE(s.memory_id, a.memory_id) AS memory_id,
                    COALESCE(s.subject_match_count, 0)::int AS subject_match_count,
                    COALESCE(a.argument_match_count, 0)::int AS argument_match_count
                FROM subject_matches s
                FULL OUTER JOIN argument_matches a ON a.memory_id = s.memory_id
            ),
            matched_entities AS (
                SELECT
                    sm.memory_id,
                    array_agg(DISTINCT qe.entity_id::text ORDER BY qe.entity_id::text) AS matched_entity_ids
                FROM semantic_memories sm
                JOIN query_entities qe
                  ON qe.entity_id = sm.subject_entity_id
                WHERE sm.status = 'ACTIVE'
                GROUP BY sm.memory_id
                UNION
                SELECT
                    ma.memory_id,
                    array_agg(DISTINCT qe.entity_id::text ORDER BY qe.entity_id::text) AS matched_entity_ids
                FROM memory_arguments ma
                JOIN semantic_memories sm ON sm.memory_id = ma.memory_id
                JOIN query_entities qe ON qe.entity_id = ma.entity_id
                WHERE sm.status = 'ACTIVE'
                  AND ma.is_entity = true
                GROUP BY ma.memory_id
            ),
            collapsed_entities AS (
                SELECT
                    memory_id,
                    array_agg(DISTINCT entity_id ORDER BY entity_id) AS matched_entity_ids
                FROM (
                    SELECT memory_id, unnest(matched_entity_ids) AS entity_id
                    FROM matched_entities
                ) x
                GROUP BY memory_id
            )
            SELECT
                m.memory_id::text AS memory_id,
                (m.subject_match_count * 3 + m.argument_match_count * 2)::double precision AS structured_score,
                m.subject_match_count,
                m.argument_match_count,
                COALESCE(ce.matched_entity_ids, ARRAY[]::text[]) AS matched_entity_ids
            FROM merged m
            LEFT JOIN collapsed_entities ce ON ce.memory_id = m.memory_id
            ORDER BY
                structured_score DESC,
                cardinality(COALESCE(ce.matched_entity_ids, ARRAY[]::text[])) DESC,
                m.memory_id ASC
            LIMIT %s
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (entity_ids, branch_k))
                rows = cur.fetchall()
        return [
            BranchCandidate(
                memory_id=row["memory_id"],
                source=SOURCE_STRUCTURED,
                rank=index,
                raw_score=float(row["structured_score"]),
                diagnostics={
                    "matched_entity_ids": list(row["matched_entity_ids"] or []),
                    "subject_match_count": int(row["subject_match_count"] or 0),
                    "argument_match_count": int(row["argument_match_count"] or 0),
                },
            )
            for index, row in enumerate(rows, start=1)
        ]

    def retrieve_graph_candidates(self, entity_ids: list[str], branch_k: int) -> list[BranchCandidate]:
        if not entity_ids:
            return []
        rows = GraphMemoryRepository(self.database_url).retrieve_candidates(
            seed_entity_ids=entity_ids,
            max_bridge_degree=get_graph_max_bridge_degree(),
            top_k=branch_k,
        )
        return [
            BranchCandidate(
                memory_id=row.memory_id,
                source=SOURCE_GRAPH,
                rank=index,
                raw_score=row.graph_signals.graph_score,
                diagnostics={
                    "graph_distance": row.graph_signals.graph_distance,
                    "direct_seed_count": row.graph_signals.direct_seed_count,
                    "bridge_path_count": row.graph_signals.bridge_path_count,
                },
            )
            for index, row in enumerate(rows, start=1)
        ]

    def hydrate_memories(self, fused_candidates: list[Any]) -> list[HydratedRetrievedMemory]:
        if not fused_candidates:
            return []
        memory_ids = [candidate.memory_id for candidate in fused_candidates]
        fused_by_id = {candidate.memory_id: candidate for candidate in fused_candidates}
        sql = """
            SELECT
                memory_id::text AS memory_id,
                canonical_text,
                status,
                version,
                subject_text,
                subject_entity_id::text AS subject_entity_id,
                subject_entity_type,
                predicate_type,
                memory_type,
                modality,
                polarity,
                certainty,
                explicitness,
                attributed_to,
                temporal_kind,
                valid_from,
                valid_to,
                event_time,
                temporal_precision,
                recurrence,
                recurrence_specifics,
                created_at
            FROM semantic_memories
            WHERE status = 'ACTIVE'
              AND memory_id = ANY(%s::uuid[])
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (memory_ids,))
                memory_rows = {row["memory_id"]: dict(row) for row in cur.fetchall()}
                cur.execute(
                    """
                    SELECT
                        memory_id::text AS memory_id,
                        role,
                        text,
                        is_entity,
                        entity_id::text AS entity_id,
                        entity_type,
                        position
                    FROM memory_arguments
                    WHERE memory_id = ANY(%s::uuid[])
                    ORDER BY memory_id, position
                    """,
                    (memory_ids,),
                )
                arguments = _group_rows(cur.fetchall(), "memory_id")
                cur.execute(
                    """
                    SELECT
                        memory_id::text AS memory_id,
                        episode_id,
                        message_id,
                        source_text,
                        observed_at
                    FROM memory_evidence
                    WHERE memory_id = ANY(%s::uuid[])
                    ORDER BY memory_id, observed_at, message_id
                    """,
                    (memory_ids,),
                )
                evidence = _group_rows(cur.fetchall(), "memory_id")

        hydrated = []
        for fused in fused_candidates:
            row = memory_rows.get(fused.memory_id)
            if row is None:
                continue
            hydrated.append(_hydrate(row, arguments.get(fused.memory_id, []), evidence.get(fused.memory_id, []), fused))
        return hydrated


def _dedupe_entity_matches(rows: list[dict[str, Any]]) -> list[QueryEntityMatch]:
    from kivi_memory.long_term_retrieval.config import QUERY_ENTITY_MIN_MARGIN

    by_text: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_text.setdefault(row["matched_text"], []).append(row)

    matches = []
    seen_entities = set()
    for matched_text, group in by_text.items():
        best_by_entity: dict[str, dict[str, Any]] = {}
        for row in group:
            existing = best_by_entity.get(row["entity_id"])
            if existing is None or _entity_match_sort_key(row) < _entity_match_sort_key(existing):
                best_by_entity[row["entity_id"]] = row

        distinct = sorted(best_by_entity.values(), key=_entity_match_sort_key)
        if len(distinct) > 1 and float(distinct[0]["score"]) - float(distinct[1]["score"]) < QUERY_ENTITY_MIN_MARGIN:
            continue
        best = distinct[0]
        if best["entity_id"] in seen_entities:
            continue
        seen_entities.add(best["entity_id"])
        matches.append(
            QueryEntityMatch(
                entity_id=best["entity_id"],
                canonical_name=best["canonical_name"],
                entity_type=best["entity_type"],
                matched_text=matched_text,
                match_method=best["match_method"],
                score=float(best["score"]),
            )
        )
    return sorted(matches, key=lambda item: (item.match_method != "exact_name", item.match_method != "exact_alias", -len(item.matched_text), item.canonical_name))


def _entity_match_sort_key(row: dict[str, Any]) -> tuple[float, int, str, str]:
    method_rank = {"exact_name": 0, "exact_alias": 1, "fuzzy": 2}.get(row["match_method"], 3)
    return (-float(row["score"]), method_rank, row["canonical_name"], row["entity_id"])


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row = _jsonable_row(dict(row))
        grouped.setdefault(row[key], []).append(row)
    return grouped


def _hydrate(row: dict[str, Any], arguments: list[dict[str, Any]], evidence: list[dict[str, Any]], fused: Any) -> HydratedRetrievedMemory:
    row = _jsonable_row(row)
    return HydratedRetrievedMemory(
        memory_id=row["memory_id"],
        canonical_text=row["canonical_text"],
        status=row["status"],
        version=int(row["version"]),
        rrf_score=fused.rrf_score,
        retrieval_sources=fused.retrieval_sources,
        branch_ranks=fused.branch_ranks,
        branch_scores=fused.branch_scores,
        subject={
            "text": row["subject_text"],
            "entity_id": row["subject_entity_id"],
            "entity_type": row["subject_entity_type"],
        },
        predicate_type=row["predicate_type"],
        memory_type=row["memory_type"],
        modality=row["modality"],
        polarity=row["polarity"],
        certainty=row["certainty"],
        explicitness=row["explicitness"],
        attributed_to=row["attributed_to"],
        temporal={
            "temporal_kind": row["temporal_kind"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "event_time": row["event_time"],
            "temporal_precision": row["temporal_precision"],
            "recurrence": row["recurrence"],
            "recurrence_specifics": row["recurrence_specifics"],
        },
        arguments=arguments,
        evidence=evidence,
        created_at=row["created_at"],
    )


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, (datetime, date)) else str(value) if isinstance(value, UUID) else value
        for key, value in row.items()
    }
