"""Minimal v1 structured retrieval over ACTIVE ledger memories."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kivi_memory.common.schemas import CandidateSemanticAssertion
from kivi_memory.entity_resolution.repository import connect
from kivi_memory.retrieval.models import StructuredMemoryCandidate, StructuredSignals

MAX_STRUCTURED_RETRIEVAL_TOP_K = 100


@dataclass(frozen=True)
class StructuredRetrievalResult:
    candidates: list[StructuredMemoryCandidate]
    db_search_ms: float
    total_ms: float


class StructuredMemoryRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def retrieve_candidates(
        self,
        *,
        subject_entity_id: str | None,
        argument_entity_ids: list[str],
        predicate_type: str,
        memory_type: str,
        top_k: int,
    ) -> list[StructuredMemoryCandidate]:
        sql = """
            SELECT
                sm.memory_id::text AS memory_id,
                sm.canonical_text,
                sm.subject_entity_id::text AS subject_entity_id,
                sm.predicate_type,
                sm.memory_type,
                sm.status,
                (sm.subject_entity_id::text = %s) AS same_subject,
                COUNT(DISTINCT ma.entity_id) FILTER (
                    WHERE ma.entity_id IS NOT NULL
                      AND ma.entity_id = ANY(%s::uuid[])
                )::int AS shared_entity_count,
                (sm.predicate_type = %s) AS same_predicate,
                (sm.memory_type = %s) AS same_memory_type
            FROM semantic_memories sm
            LEFT JOIN memory_arguments ma ON ma.memory_id = sm.memory_id
            WHERE sm.status = 'ACTIVE'
              AND (
                (%s::text IS NOT NULL AND sm.subject_entity_id::text = %s)
                OR (cardinality(%s::uuid[]) > 0 AND ma.entity_id = ANY(%s::uuid[]))
                OR sm.predicate_type = %s
                OR sm.memory_type = %s
              )
            GROUP BY sm.memory_id
            ORDER BY
                (sm.subject_entity_id::text = %s) DESC,
                COUNT(DISTINCT ma.entity_id) FILTER (
                    WHERE ma.entity_id IS NOT NULL
                      AND ma.entity_id = ANY(%s::uuid[])
                ) DESC,
                (sm.predicate_type = %s) DESC,
                (sm.memory_type = %s) DESC,
                sm.updated_at DESC
            LIMIT %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        subject_entity_id,
                        argument_entity_ids,
                        predicate_type,
                        memory_type,
                        subject_entity_id,
                        subject_entity_id,
                        argument_entity_ids,
                        argument_entity_ids,
                        predicate_type,
                        memory_type,
                        subject_entity_id,
                        argument_entity_ids,
                        predicate_type,
                        memory_type,
                        top_k,
                    ),
                )
                rows = cur.fetchall()
        return [
            StructuredMemoryCandidate(
                memory_id=row["memory_id"],
                canonical_text=row["canonical_text"],
                structured_signals=StructuredSignals(
                    same_subject=bool(row["same_subject"]),
                    shared_entity_count=int(row["shared_entity_count"] or 0),
                    same_predicate=bool(row["same_predicate"]),
                    same_memory_type=bool(row["same_memory_type"]),
                ),
                subject_entity_id=row["subject_entity_id"],
                predicate_type=row["predicate_type"],
                memory_type=row["memory_type"],
                status=row["status"],
            )
            for row in rows
        ]


def retrieve_structured_candidates(
    assertion: CandidateSemanticAssertion | dict[str, Any],
    entity_resolution: dict[str, Any],
    top_k: int = 20,
    repository: StructuredMemoryRepository | None = None,
) -> StructuredRetrievalResult:
    if top_k < 1 or top_k > MAX_STRUCTURED_RETRIEVAL_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_STRUCTURED_RETRIEVAL_TOP_K}")

    assertion = CandidateSemanticAssertion.model_validate(assertion)
    repo = repository or StructuredMemoryRepository()
    total_started = time.perf_counter()
    db_started = time.perf_counter()
    candidates = repo.retrieve_candidates(
        subject_entity_id=_resolved_subject_id(entity_resolution),
        argument_entity_ids=_resolved_argument_ids(assertion, entity_resolution),
        predicate_type=assertion.predicate_type,
        memory_type=_enum_value(assertion.memory_type),
        top_k=top_k,
    )
    db_ms = (time.perf_counter() - db_started) * 1000
    return StructuredRetrievalResult(
        candidates=candidates,
        db_search_ms=db_ms,
        total_ms=(time.perf_counter() - total_started) * 1000,
    )


def _resolved_subject_id(entity_resolution: dict[str, Any]) -> str | None:
    subject = entity_resolution.get("subject") or {}
    if subject.get("resolution") in {"MATCHED", "NEW"}:
        return subject.get("entity_id")
    return None


def _resolved_argument_ids(
    assertion: CandidateSemanticAssertion,
    entity_resolution: dict[str, Any],
) -> list[str]:
    ids = []
    resolved_arguments = iter(entity_resolution.get("semantic_arguments") or [])
    for argument in assertion.semantic_arguments:
        result = next(resolved_arguments, {}).get("result") if argument.is_entity else None
        if result and result.get("resolution") in {"MATCHED", "NEW"} and result.get("entity_id"):
            ids.append(str(result["entity_id"]))
    return sorted(set(ids))


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value
