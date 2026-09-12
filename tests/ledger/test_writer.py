from __future__ import annotations

from datetime import datetime

import pytest

from kivi_memory.common.schemas import (
    CandidateSemanticAssertion,
    EntityMention,
    EpisodeMessage,
    MemoryEpisode,
    MemoryType,
    Modality,
    Polarity,
    SourceSpan,
    TemporalKind,
    TemporalMetadata,
    TemporalPrecision,
    Recurrence,
    SemanticArgument,
    ValidationReport,
    ValidationSeverity,
    ValidationIssue,
    Certainty,
    Explicitness,
)
from kivi_memory.ledger.models import AddMemoryResult
from kivi_memory.ledger.writer import add_memory


class CapturingRepository:
    def __init__(self) -> None:
        self.memory_values = None
        self.argument_values = None
        self.evidence_values = None
        self.event_payload = None

    def add_memory_record(self, *, memory_values, argument_values, evidence_values, event_payload):
        self.memory_values = memory_values
        self.argument_values = argument_values
        self.evidence_values = evidence_values
        self.event_payload = event_payload
        return AddMemoryResult(memory_id="memory-1", status="ACTIVE", version=1)


def test_successful_insert_writes_all_ledger_payloads() -> None:
    repo = CapturingRepository()

    result = add_memory(
        assertion(),
        temporal_metadata(),
        entity_resolution(),
        episode(),
        sensitivity_metadata=None,
        validation_report=valid_report(),
        repository=repo,
    )

    assert result.memory_id == "memory-1"
    assert repo.memory_values["canonical_text"] == "The user met Kavya yesterday."
    assert len(repo.argument_values) == 3
    assert len(repo.evidence_values) == 1
    assert repo.event_payload == {
        "version": 1,
        "status": "ACTIVE",
        "source_episode_id": "ep-1",
    }


def test_subject_entity_id_stored_correctly() -> None:
    repo = CapturingRepository()

    add_memory(assertion(), temporal_metadata(), entity_resolution(), episode(), validation_report=valid_report(), repository=repo)

    assert repo.memory_values["subject_entity_id"] == "user-1"


def test_resolved_argument_entity_id_stored_correctly() -> None:
    repo = CapturingRepository()

    add_memory(assertion(), temporal_metadata(), entity_resolution(), episode(), validation_report=valid_report(), repository=repo)

    assert repo.argument_values[0]["entity_id"] == "kavya-1"
    assert repo.argument_values[0]["is_entity"] is True


def test_ambiguous_argument_keeps_text_with_null_entity_id() -> None:
    repo = CapturingRepository()

    add_memory(assertion(), temporal_metadata(), entity_resolution(), episode(), validation_report=valid_report(), repository=repo)

    assert repo.argument_values[1]["text"] == "Orion"
    assert repo.argument_values[1]["entity_id"] is None


def test_non_entity_argument_stores_null_entity_id() -> None:
    repo = CapturingRepository()

    add_memory(assertion(), temporal_metadata(), entity_resolution(), episode(), validation_report=valid_report(), repository=repo)

    assert repo.argument_values[2]["is_entity"] is False
    assert repo.argument_values[2]["entity_id"] is None


def test_source_span_timestamp_maps_to_observed_at() -> None:
    repo = CapturingRepository()

    add_memory(assertion(), temporal_metadata(), entity_resolution(), episode(), validation_report=valid_report(), repository=repo)

    assert repo.evidence_values[0]["observed_at"] == datetime.fromisoformat("2026-09-10T09:00:00+05:30")


def test_invalid_assertion_is_rejected() -> None:
    with pytest.raises(ValueError, match="validation_report.valid == true"):
        add_memory(assertion(), temporal_metadata(), entity_resolution(), episode(), validation_report=invalid_report())


def test_missing_source_message_fails_cleanly() -> None:
    with pytest.raises(ValueError, match="source message ID not found"):
        add_memory(assertion(), temporal_metadata(), entity_resolution(), episode_without_source(), validation_report=valid_report())


def assertion() -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion(
        canonical_text="The user met Kavya yesterday.",
        memory_type=MemoryType.IMPORTANT_EVENT,
        subject=EntityMention(text="the user", entity_type="person"),
        semantic_arguments=[
            SemanticArgument(role="person", text="Kavya", is_entity=True, entity_type="person"),
            SemanticArgument(role="project", text="Orion", is_entity=True, entity_type="project"),
            SemanticArgument(role="time", text="yesterday", is_entity=False, entity_type=None),
        ],
        predicate_type="MET",
        modality=Modality.FACT,
        polarity=Polarity.POSITIVE,
        certainty=Certainty.CERTAIN,
        explicitness=Explicitness.EXPLICIT,
        attributed_to="user",
        source_spans=[SourceSpan(message_id="m1", text="I met Kavya yesterday.")],
    )


def temporal_metadata() -> TemporalMetadata:
    return TemporalMetadata(
        temporal_kind=TemporalKind.DISCRETE_EVENT,
        valid_from_hint=None,
        valid_to_hint=None,
        event_time="2026-09-09T09:00:00+05:30",
        temporal_precision=TemporalPrecision.DAY,
        recurrence=Recurrence.NONE,
        recurrence_specifics=None,
    )


def entity_resolution() -> dict:
    return {
        "subject": {"resolution": "MATCHED", "entity_id": "user-1"},
        "semantic_arguments": [
            {"role": "person", "result": {"resolution": "MATCHED", "entity_id": "kavya-1"}},
            {"role": "project", "result": {"resolution": "AMBIGUOUS", "entity_id": None}},
        ],
    }


def episode() -> MemoryEpisode:
    return MemoryEpisode(
        episode_id="ep-1",
        messages=[
            EpisodeMessage(
                message_id="m1",
                role="USER",
                timestamp="2026-09-10T09:00:00+05:30",
                text="I met Kavya yesterday.",
            )
        ],
    )


def episode_without_source() -> MemoryEpisode:
    return MemoryEpisode(
        episode_id="ep-1",
        messages=[
            EpisodeMessage(
                message_id="m2",
                role="USER",
                timestamp="2026-09-10T09:00:00+05:30",
                text="Different message.",
            )
        ],
    )


def valid_report() -> ValidationReport:
    return ValidationReport(valid=True, issues=[])


def invalid_report() -> ValidationReport:
    return ValidationReport(
        valid=False,
        issues=[
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="TEST_ERROR",
                message="invalid",
                assertion_index=0,
            )
        ],
    )
