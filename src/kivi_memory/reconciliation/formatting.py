"""Compact deterministic formatting for the reconciliation judge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from kivi_memory.common.schemas import CandidateSemanticAssertion, TemporalMetadata
from kivi_memory.entity_resolution.normalizer import normalize_entity_name
from kivi_memory.reconciliation.models import HydratedMemory

COMPACT_TEMPORAL_TIMEZONE = ZoneInfo("Asia/Kolkata")


@dataclass
class EntityTokenMapper:
    """Maps real entity UUIDs to stable short E tokens within one request."""

    _tokens: dict[str, str]

    def __init__(self) -> None:
        self._tokens = {}

    def token_for(self, entity_id: str | None) -> str:
        if not entity_id:
            return "-"
        entity_id = str(entity_id)
        if entity_id not in self._tokens:
            self._tokens[entity_id] = f"E{len(self._tokens) + 1}"
        return self._tokens[entity_id]


@dataclass(frozen=True)
class CompactIncoming:
    assertion: CandidateSemanticAssertion
    subject_entity_id: str | None
    argument_entities: list[tuple[str, str]]
    temporal_signature: str


def build_compact_input(
    incoming_assertion: CandidateSemanticAssertion | dict[str, Any],
    entity_resolution: dict[str, Any] | None,
    temporal_metadata: TemporalMetadata | dict[str, Any] | None,
    hydrated_candidates: list[HydratedMemory],
) -> tuple[str, dict[str, str], CompactIncoming]:
    assertion = CandidateSemanticAssertion.model_validate(_to_plain(incoming_assertion))
    temporal = TemporalMetadata.model_validate(_to_plain(temporal_metadata)) if temporal_metadata else None
    incoming = CompactIncoming(
        assertion=assertion,
        subject_entity_id=_incoming_subject_entity_id(entity_resolution),
        argument_entities=_incoming_argument_entities(assertion, entity_resolution),
        temporal_signature=format_temporal(temporal),
    )
    mapper = EntityTokenMapper()
    lines = [_format_incoming(incoming, mapper)]
    candidate_map = {}
    for index, candidate in enumerate(hydrated_candidates, start=1):
        candidate_id = f"C{index}"
        candidate_map[candidate_id] = candidate.memory_id
        lines.append(_format_candidate(candidate_id, candidate, mapper))
    return "\n".join(lines), candidate_map, incoming


def semantic_signature_for_incoming(incoming: CompactIncoming) -> tuple[Any, ...]:
    assertion = incoming.assertion
    return (
        normalize_entity_name(assertion.canonical_text),
        incoming.subject_entity_id,
        assertion.predicate_type,
        tuple(sorted(incoming.argument_entities)),
        _enum_value(assertion.polarity),
        _enum_value(assertion.modality),
        incoming.temporal_signature,
    )


def semantic_signature_for_memory(memory: HydratedMemory) -> tuple[Any, ...]:
    return (
        normalize_entity_name(memory.canonical_text),
        memory.subject_entity_id,
        memory.predicate_type,
        tuple(sorted((argument.role, argument.entity_id) for argument in memory.arguments if argument.is_entity and argument.entity_id)),
        memory.polarity,
        memory.modality,
        format_temporal(memory),
    )


def format_temporal(value: TemporalMetadata | HydratedMemory | None) -> str:
    if value is None:
        return "-"

    kind = _get(value, "temporal_kind")
    precision = _get(value, "temporal_precision")
    recurrence = _get(value, "recurrence")
    recurrence_specifics = _get(value, "recurrence_specifics")
    event_time = _get(value, "event_time")
    valid_from = _get(value, "valid_from_hint") or _get(value, "valid_from")
    valid_to = _get(value, "valid_to_hint") or _get(value, "valid_to")

    if not kind or kind == "NONE":
        return "-"
    if kind == "DISCRETE_EVENT" and event_time:
        return f"E@{_short_time(event_time)};pr={precision or '-'}"
    if kind == "STATE_INTERVAL" and (valid_from or valid_to):
        return f"S[{_short_time(valid_from) if valid_from else ''}..{_short_time(valid_to) if valid_to else ''}];pr={precision or '-'}"
    if kind == "RECURRENCE" and recurrence and recurrence != "NONE":
        suffix = f":{recurrence_specifics}" if recurrence_specifics else ""
        return f"R:{recurrence}{suffix};pr={precision or '-'}"
    return "-"


def _format_incoming(incoming: CompactIncoming, mapper: EntityTokenMapper) -> str:
    assertion = incoming.assertion
    return _compact_line(
        "I",
        canonical_text=assertion.canonical_text,
        subject=mapper.token_for(incoming.subject_entity_id),
        predicate=assertion.predicate_type,
        arguments=_format_argument_entities(incoming.argument_entities, mapper),
        modality=_enum_value(assertion.modality),
        polarity=_enum_value(assertion.polarity),
        certainty=_enum_value(assertion.certainty),
        temporal=incoming.temporal_signature,
    )


def _format_candidate(label: str, memory: HydratedMemory, mapper: EntityTokenMapper) -> str:
    argument_entities = [(argument.role, argument.entity_id) for argument in memory.arguments if argument.is_entity and argument.entity_id]
    return _compact_line(
        label,
        canonical_text=memory.canonical_text,
        subject=mapper.token_for(memory.subject_entity_id),
        predicate=memory.predicate_type,
        arguments=_format_argument_entities(argument_entities, mapper),
        modality=memory.modality,
        polarity=memory.polarity,
        certainty=memory.certainty,
        temporal=format_temporal(memory),
    )


def _compact_line(
    label: str,
    *,
    canonical_text: str,
    subject: str,
    predicate: str,
    arguments: str,
    modality: str,
    polarity: str,
    certainty: str,
    temporal: str,
) -> str:
    return (
        f"{label}|x={_compact_text(canonical_text)}|s={subject}|p={predicate}|"
        f"a={arguments}|m={modality}|pol={polarity}|c={certainty}|t={temporal}"
    )


def _format_argument_entities(argument_entities: list[tuple[str, str]], mapper: EntityTokenMapper) -> str:
    if not argument_entities:
        return "-"
    return ",".join(f"{role}:{mapper.token_for(entity_id)}" for role, entity_id in argument_entities)


def _incoming_subject_entity_id(entity_resolution: dict[str, Any] | None) -> str | None:
    subject = (entity_resolution or {}).get("subject") or {}
    if subject.get("resolution") in {"MATCHED", "NEW"}:
        return subject.get("entity_id")
    return None


def _incoming_argument_entities(
    assertion: CandidateSemanticAssertion,
    entity_resolution: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    resolved_arguments = iter((entity_resolution or {}).get("semantic_arguments") or [])
    entities = []
    for argument in assertion.semantic_arguments:
        result = next(resolved_arguments, {}).get("result") if argument.is_entity else None
        if result and result.get("resolution") in {"MATCHED", "NEW"} and result.get("entity_id"):
            entities.append((argument.role, str(result["entity_id"])))
    return entities


def _short_time(value: Any) -> str:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            text = parsed.astimezone(COMPACT_TEMPORAL_TIMEZONE).isoformat()
    except ValueError:
        pass
    match = re.match(r"^\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)
    match = re.match(r"^\d{4}-\d{2}", text)
    if match:
        return match.group(0)
    return text


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value
