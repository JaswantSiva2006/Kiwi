"""Public writer for the add-only Canonical Memory Ledger."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from kivi_memory.common.schemas import (
    CandidateSemanticAssertion,
    MemoryEpisode,
    TemporalMetadata,
    ValidationReport,
)
from kivi_memory.ledger.models import AddMemoryResult, StoredMemory
from kivi_memory.ledger.repository import LedgerRepository


def add_memory(
    assertion: CandidateSemanticAssertion | dict[str, Any],
    temporal_metadata: TemporalMetadata | dict[str, Any] | None,
    entity_resolution: dict[str, Any],
    episode: MemoryEpisode,
    sensitivity_metadata: Any = None,
    validation_report: ValidationReport | dict[str, Any] | None = None,
    repository: LedgerRepository | None = None,
) -> AddMemoryResult:
    """Persist one valid resolved/enriched assertion into the ledger tables."""

    values = build_memory_insert_values(
        assertion=assertion,
        temporal_metadata=temporal_metadata,
        entity_resolution=entity_resolution,
        episode=episode,
        sensitivity_metadata=sensitivity_metadata,
        validation_report=validation_report,
    )
    repo = repository or LedgerRepository()
    return repo.add_memory_record(
        memory_values=values["memory_values"],
        argument_values=values["argument_values"],
        evidence_values=values["evidence_values"],
        event_payload=values["event_payload"],
    )


def build_memory_insert_values(
    *,
    assertion: CandidateSemanticAssertion | dict[str, Any],
    temporal_metadata: TemporalMetadata | dict[str, Any] | None,
    entity_resolution: dict[str, Any],
    episode: MemoryEpisode,
    sensitivity_metadata: Any = None,
    validation_report: ValidationReport | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare canonical ledger insert values without opening a transaction."""

    del sensitivity_metadata
    assertion = CandidateSemanticAssertion.model_validate(_to_plain(assertion))
    temporal_metadata = TemporalMetadata.model_validate(_to_plain(temporal_metadata)) if temporal_metadata else None
    validation_report = ValidationReport.model_validate(_to_plain(validation_report)) if validation_report else None
    if validation_report is None or not validation_report.valid:
        raise ValueError("add_memory requires validation_report.valid == true")

    messages_by_id = {message.message_id: message for message in episode.messages}
    evidence_values = []
    for span in assertion.source_spans:
        message = messages_by_id.get(span.message_id)
        if message is None:
            raise ValueError(f"source message ID not found: {span.message_id}")
        evidence_values.append(
            {
                "episode_id": episode.episode_id,
                "message_id": span.message_id,
                "source_text": span.text,
                "observed_at": _parse_timestamp(message.timestamp),
            }
        )

    return {
        "memory_values": _memory_values(assertion, temporal_metadata, entity_resolution),
        "argument_values": _argument_values(assertion, entity_resolution),
        "evidence_values": evidence_values,
        "event_payload": {
            "version": 1,
            "status": "ACTIVE",
            "source_episode_id": episode.episode_id,
        },
    }


def get_memory(memory_id: str, repository: LedgerRepository | None = None) -> StoredMemory:
    repo = repository or LedgerRepository()
    return repo.get_memory(memory_id)


def list_memories(
    limit: int = 50,
    status: str | None = None,
    memory_type: str | None = None,
    subject_entity_id: str | None = None,
    repository: LedgerRepository | None = None,
) -> list[dict[str, Any]]:
    repo = repository or LedgerRepository()
    return repo.list_memories(limit, status, memory_type, subject_entity_id)


def _memory_values(
    assertion: CandidateSemanticAssertion,
    temporal_metadata: TemporalMetadata | None,
    entity_resolution: dict[str, Any],
) -> dict[str, Any]:
    subject_resolution = entity_resolution.get("subject") or {}
    return {
        "canonical_text": assertion.canonical_text,
        "subject_entity_id": _resolved_entity_id(subject_resolution),
        "subject_text": assertion.subject.text,
        "subject_entity_type": assertion.subject.entity_type,
        "predicate_type": assertion.predicate_type,
        "memory_type": _enum_value(assertion.memory_type),
        "modality": _enum_value(assertion.modality),
        "polarity": _enum_value(assertion.polarity),
        "certainty": _enum_value(assertion.certainty),
        "explicitness": _enum_value(assertion.explicitness),
        "attributed_to": assertion.attributed_to,
        "temporal_kind": _enum_value(temporal_metadata.temporal_kind) if temporal_metadata else None,
        "valid_from": temporal_metadata.valid_from_hint if temporal_metadata else None,
        "valid_to": temporal_metadata.valid_to_hint if temporal_metadata else None,
        "event_time": temporal_metadata.event_time if temporal_metadata else None,
        "temporal_precision": _enum_value(temporal_metadata.temporal_precision) if temporal_metadata else None,
        "recurrence": _enum_value(temporal_metadata.recurrence) if temporal_metadata else None,
        "recurrence_specifics": temporal_metadata.recurrence_specifics if temporal_metadata else None,
    }


def _argument_values(
    assertion: CandidateSemanticAssertion,
    entity_resolution: dict[str, Any],
) -> list[dict[str, Any]]:
    resolved_arguments = iter(entity_resolution.get("semantic_arguments") or [])
    values = []
    for position, argument in enumerate(assertion.semantic_arguments):
        argument_resolution = next(resolved_arguments, {}).get("result") if argument.is_entity else None
        values.append(
            {
                "role": argument.role,
                "text": argument.text,
                "is_entity": argument.is_entity,
                "entity_type": argument.entity_type,
                "entity_id": _resolved_entity_id(argument_resolution),
                "position": position,
            }
        )
    return values


def _resolved_entity_id(resolution: dict[str, Any] | None) -> str | None:
    if not resolution:
        return None
    if resolution.get("resolution") not in {"MATCHED", "NEW"}:
        return None
    return resolution.get("entity_id")


def _parse_timestamp(value: datetime | str) -> datetime | str:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
