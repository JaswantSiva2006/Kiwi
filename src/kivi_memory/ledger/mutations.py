"""Deterministic execution of validated reconciliation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from kivi_memory.calendar import (
    cancel_calendar_event_for_memory,
    create_calendar_event_for_memory,
    mark_calendar_event_superseded,
)
from kivi_memory.common.schemas import MemoryEpisode
from kivi_memory.ledger.models import AddMemoryResult
from kivi_memory.ledger.repository import LedgerRepository, insert_memory_record_in_transaction
from kivi_memory.ledger.writer import build_memory_insert_values
from kivi_memory.projections import run_post_commit_projections
from kivi_memory.reconciliation.models import ReconciliationDecision, ReconciliationOp

TARGETED_OPS = {
    ReconciliationOp.REINFORCE,
    ReconciliationOp.SUPERSEDE,
    ReconciliationOp.RETRACT,
}
TARGET_STATUSES = {
    ReconciliationOp.SUPERSEDE: "SUPERSEDED",
    ReconciliationOp.RETRACT: "RETRACTED",
}
TARGET_EVENTS = {
    ReconciliationOp.REINFORCE: "MEMORY_REINFORCED",
    ReconciliationOp.SUPERSEDE: "MEMORY_SUPERSEDED",
    ReconciliationOp.RETRACT: "MEMORY_RETRACTED",
}


@dataclass(frozen=True)
class LedgerMutationResult:
    op: ReconciliationOp
    target_memory_ids: list[str]
    created_memory_id: str | None
    executed: bool
    event_type: str | None
    embedding_projection: Any = None
    embedding_projection_error: str | None = None
    graph_projection: Any = None
    graph_projection_error: str | None = None
    calendar_event_id: str | None = None
    calendar_status_updates: list[dict[str, Any]] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "target_memory_ids": self.target_memory_ids,
            "created_memory_id": self.created_memory_id,
            "executed": self.executed,
            "event_type": self.event_type,
            "embedding_projection": _to_plain(self.embedding_projection),
            "embedding_projection_error": self.embedding_projection_error,
            "graph_projection": _to_plain(self.graph_projection),
            "graph_projection_error": self.graph_projection_error,
            "calendar_event_id": self.calendar_event_id,
            "calendar_status_updates": self.calendar_status_updates or [],
        }


Projection = Callable[[str], Any]


def execute_reconciliation(
    decision: ReconciliationDecision | dict[str, Any],
    incoming_assertion: Any,
    final_candidates: list[Any],
    temporal_metadata: Any,
    entity_resolution: dict[str, Any],
    episode: MemoryEpisode,
    validation_report: Any = None,
    repository: LedgerRepository | None = None,
    indexer: Projection | None = None,
    graph_projector: Projection | None = None,
) -> LedgerMutationResult:
    """Apply a reconciliation decision to the canonical ledger.

    The operation is deterministic and contains no LLM calls. Canonical ledger
    writes commit before any embedding projection update is attempted.
    """

    incoming_assertion, temporal_metadata, entity_resolution, validation_report = _unwrap_incoming(
        incoming_assertion,
        temporal_metadata,
        entity_resolution,
        validation_report,
    )
    op, requested_targets = _decision_parts(decision)
    ranked_ids = _first_ranked_memory_ids(final_candidates)
    target_ids = _resolve_targets(op, requested_targets, ranked_ids)

    if op == ReconciliationOp.NO_MEMORY:
        return LedgerMutationResult(op, [], None, False, None)

    repo = repository or LedgerRepository()
    projection_memory_id = None
    event_type = None
    calendar_event_id = None
    calendar_status_updates: list[dict[str, Any]] = []
    with repo.connect() as conn:
        with conn.cursor() as cur:
            insert_values = None
            if op in {ReconciliationOp.ADD, ReconciliationOp.SUPERSEDE, ReconciliationOp.RETRACT}:
                insert_values = build_memory_insert_values(
                    assertion=incoming_assertion,
                    temporal_metadata=temporal_metadata,
                    entity_resolution=entity_resolution,
                    episode=episode,
                    validation_report=validation_report,
                )

            if op == ReconciliationOp.ADD:
                created = _insert_new_memory(cur, insert_values)
                projection_memory_id = created.memory_id
                calendar_event_id = create_calendar_event_for_memory(
                    cur,
                    memory_id=created.memory_id,
                    assertion=incoming_assertion,
                    temporal_metadata=temporal_metadata,
                    entity_resolution=entity_resolution,
                    episode=episode,
                )
                event_type = "MEMORY_ADDED"

            elif op == ReconciliationOp.REINFORCE:
                target_id = target_ids[0]
                repo.lock_active_memory(cur, target_id)
                evidence_values = build_memory_insert_values(
                    assertion=incoming_assertion,
                    temporal_metadata=temporal_metadata,
                    entity_resolution=entity_resolution,
                    episode=episode,
                    validation_report=validation_report,
                )["evidence_values"]
                inserted_count = repo.append_evidence_in_transaction(
                    cur,
                    memory_id=target_id,
                    evidence_values=evidence_values,
                )
                repo.increment_memory_version_in_transaction(cur, memory_id=target_id)
                event_type = TARGET_EVENTS[op]
                repo.insert_event_in_transaction(
                    cur,
                    memory_id=target_id,
                    event_type=event_type,
                    payload={
                        "source_episode_id": episode.episode_id,
                        "evidence_inserted": inserted_count,
                    },
                )

            elif op in {ReconciliationOp.SUPERSEDE, ReconciliationOp.RETRACT}:
                target_id = target_ids[0]
                repo.lock_active_memory(cur, target_id)
                created = _insert_new_memory(cur, insert_values)
                projection_memory_id = created.memory_id
                repo.update_memory_status_in_transaction(
                    cur,
                    memory_id=target_id,
                    status=TARGET_STATUSES[op],
                )
                if op == ReconciliationOp.SUPERSEDE:
                    calendar_status_updates.append(
                        {
                            "memory_id": target_id,
                            "status": "SUPERSEDED",
                            "updated": mark_calendar_event_superseded(cur, memory_id=target_id),
                        }
                    )
                    calendar_event_id = create_calendar_event_for_memory(
                        cur,
                        memory_id=created.memory_id,
                        assertion=incoming_assertion,
                        temporal_metadata=temporal_metadata,
                        entity_resolution=entity_resolution,
                        episode=episode,
                    )
                else:
                    calendar_status_updates.append(
                        {
                            "memory_id": target_id,
                            "status": "CANCELLED",
                            "updated": cancel_calendar_event_for_memory(cur, memory_id=target_id),
                        }
                    )
                event_type = TARGET_EVENTS[op]
                payload_key = "replacement_memory_id" if op == ReconciliationOp.SUPERSEDE else "correction_memory_id"
                repo.insert_event_in_transaction(
                    cur,
                    memory_id=target_id,
                    event_type=event_type,
                    payload={
                        payload_key: created.memory_id,
                        "source_episode_id": episode.episode_id,
                    },
                )

    projections = None
    if projection_memory_id is not None:
        projections = run_post_commit_projections(
            projection_memory_id,
            graph_projector=graph_projector,
            embedding_indexer=indexer,
        )

    return LedgerMutationResult(
        op=op,
        target_memory_ids=target_ids,
        created_memory_id=projection_memory_id,
        executed=op != ReconciliationOp.NO_MEMORY,
        event_type=event_type,
        embedding_projection=_get_projection(projections, "embedding_projection"),
        embedding_projection_error=_get_projection(projections, "embedding_projection_error"),
        graph_projection=_get_projection(projections, "graph_projection"),
        graph_projection_error=_get_projection(projections, "graph_projection_error"),
        calendar_event_id=calendar_event_id,
        calendar_status_updates=calendar_status_updates,
    )


def _insert_new_memory(cur, insert_values: dict[str, Any] | None) -> AddMemoryResult:
    if insert_values is None:
        raise RuntimeError("missing insert values")
    return insert_memory_record_in_transaction(
        cur,
        memory_values=insert_values["memory_values"],
        argument_values=insert_values["argument_values"],
        evidence_values=insert_values["evidence_values"],
        event_payload=insert_values["event_payload"],
    )


def _get_projection(projections: Any, key: str) -> Any:
    return getattr(projections, key) if projections is not None else None


def _unwrap_incoming(
    incoming_assertion: Any,
    temporal_metadata: Any,
    entity_resolution: dict[str, Any] | None,
    validation_report: Any,
) -> tuple[Any, Any, dict[str, Any], Any]:
    if isinstance(incoming_assertion, dict) and "semantic_assertion" in incoming_assertion:
        wrapper = incoming_assertion
        return (
            wrapper["semantic_assertion"],
            temporal_metadata if temporal_metadata is not None else wrapper.get("temporal_metadata"),
            entity_resolution if entity_resolution is not None else wrapper.get("entity_resolution"),
            validation_report if validation_report is not None else wrapper.get("validation_report"),
        )
    if entity_resolution is None:
        raise ValueError("entity_resolution is required")
    return incoming_assertion, temporal_metadata, entity_resolution, validation_report


def _decision_parts(decision: ReconciliationDecision | dict[str, Any]) -> tuple[ReconciliationOp, list[str]]:
    if isinstance(decision, ReconciliationDecision):
        return decision.op, list(decision.target_memory_ids)
    op = ReconciliationOp(_enum_value(decision["op"]))
    targets = decision.get("target_memory_ids", decision.get("targets", []))
    return op, [str(target) for target in targets]


def _resolve_targets(op: ReconciliationOp, requested_targets: list[str], ranked_ids: list[str]) -> list[str]:
    c_map = {f"C{index}": memory_id for index, memory_id in enumerate(ranked_ids, start=1)}
    if op in {ReconciliationOp.ADD, ReconciliationOp.NO_MEMORY}:
        if requested_targets:
            raise ValueError(f"{op.value} requires no targets")
        return []
    if op in TARGETED_OPS and len(requested_targets) != 1:
        raise ValueError(f"{op.value} requires exactly one target")
    target = requested_targets[0]
    if target in c_map:
        return [c_map[target]]
    if target in ranked_ids:
        return [target]
    raise ValueError(f"invalid reconciliation target: {target}")


def _first_ranked_memory_ids(final_candidates: list[Any]) -> list[str]:
    memory_ids = []
    seen = set()
    for candidate in final_candidates[:8]:
        memory_id = _get(candidate, "memory_id")
        if memory_id is None:
            continue
        memory_id = str(memory_id)
        if memory_id and memory_id not in seen:
            seen.add(memory_id)
            memory_ids.append(memory_id)
    return memory_ids


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return value
