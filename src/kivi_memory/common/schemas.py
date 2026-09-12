"""Pydantic contracts for semantic compiler input and output."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EpisodeRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"
    SYSTEM = "SYSTEM"


class MemoryType(StrEnum):
    CALENDAR_EVENT = "CALENDAR_EVENT"
    PEOPLE_RELATIONSHIP = "PEOPLE_RELATIONSHIP"
    PROJECT_GOAL_TOPIC = "PROJECT_GOAL_TOPIC"
    PREFERENCE = "PREFERENCE"
    HABIT_ROUTINE = "HABIT_ROUTINE"
    WORKFLOW = "WORKFLOW"
    DECISION = "DECISION"
    COMMITMENT_OPEN_LOOP = "COMMITMENT_OPEN_LOOP"
    PERSONAL_MEANING_ALIAS = "PERSONAL_MEANING_ALIAS"
    ENTITY_CONTEXT = "ENTITY_CONTEXT"
    PERSONAL_CONTEXT = "PERSONAL_CONTEXT"
    STANDING_RULE = "STANDING_RULE"
    IMPORTANT_EVENT = "IMPORTANT_EVENT"
    OTHER = "OTHER"


class Modality(StrEnum):
    FACT = "FACT"
    PLAN = "PLAN"
    GOAL = "GOAL"
    INTENTION = "INTENTION"
    BELIEF = "BELIEF"
    PREFERENCE = "PREFERENCE"
    OBLIGATION = "OBLIGATION"
    HYPOTHETICAL = "HYPOTHETICAL"
    PREDICTION = "PREDICTION"
    DESIRE = "DESIRE"


class Polarity(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class Certainty(StrEnum):
    CERTAIN = "CERTAIN"
    PROBABLE = "PROBABLE"
    TENTATIVE = "TENTATIVE"
    POSSIBLE = "POSSIBLE"
    UNCERTAIN = "UNCERTAIN"


class Explicitness(StrEnum):
    EXPLICIT = "EXPLICIT"
    IMPLICIT = "IMPLICIT"
    INFERRED = "INFERRED"
    QUOTED = "QUOTED"


class TemporalPrecision(StrEnum):
    NONE = "NONE"
    EXACT = "EXACT"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    YEAR = "YEAR"
    APPROXIMATE = "APPROXIMATE"


class Recurrence(StrEnum):
    NONE = "NONE"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


class TemporalKind(StrEnum):
    NONE = "NONE"
    STATE_INTERVAL = "STATE_INTERVAL"
    DISCRETE_EVENT = "DISCRETE_EVENT"
    RECURRENCE = "RECURRENCE"


class ValidationSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"


IssueSeverity = ValidationSeverity


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpisodeMessage(StrictBaseModel):
    message_id: str
    role: EpisodeRole
    timestamp: datetime | str
    text: str
    memory_eligible: bool | None = None
    context_only: bool | None = None
    source_thread_episode_id: str | None = None
    source_stream_id: str | None = None

    @field_validator("message_id", "text")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("source_thread_episode_id", "source_stream_id")
    @classmethod
    def optional_non_empty_string(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be null or non-empty")
        return value


class ToolContextItem(StrictBaseModel):
    tool_name: str
    content: str
    source_message_id: str | None = None

    @field_validator("tool_name", "content")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class GroundingContext(StrictBaseModel):
    current_app: str | None = None
    active_document: str | None = None
    active_participant: str | None = None
    selected_text: str | None = None


class MemoryEpisode(StrictBaseModel):
    episode_id: str
    session_id: str | None = None
    timezone: str | None = None
    locale: str | None = None
    messages: list[EpisodeMessage] = Field(min_length=1)
    tool_context: list[ToolContextItem] | None = None
    grounding_context: GroundingContext | None = None

    @field_validator("episode_id")
    @classmethod
    def episode_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_messages(self) -> MemoryEpisode:
        ids = [message.message_id for message in self.messages]
        if len(ids) != len(set(ids)):
            raise ValueError("message IDs must be unique inside the episode")

        if not any(message.role == EpisodeRole.USER for message in self.messages):
            raise ValueError("at least one USER message must exist")

        parsed_timestamps: list[datetime] = []
        for message in self.messages:
            parsed = _parse_timestamp(message.timestamp)
            if parsed is None:
                parsed_timestamps = []
                break
            parsed_timestamps.append(parsed)

        if parsed_timestamps and parsed_timestamps != sorted(parsed_timestamps):
            raise ValueError("messages must be ordered by timestamp")

        return self


class EntityMention(StrictBaseModel):
    text: str
    entity_type: str

    @field_validator("text", "entity_type")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class SemanticArgument(StrictBaseModel):
    role: str
    text: str
    is_entity: bool
    entity_type: str | None

    @field_validator("role", "text")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_entity_fields(self) -> SemanticArgument:
        if not self.is_entity and self.entity_type is not None:
            raise ValueError("entity_type must be null when is_entity is false")
        return self


class SourceSpan(StrictBaseModel):
    message_id: str
    text: str

    @field_validator("message_id", "text")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class CalendarEventExtraction(StrictBaseModel):
    title: str
    event_kind: str | None = None
    location_text: str | None = None
    start_time_text: str | None = None
    end_time_text: str | None = None
    duration_text: str | None = None
    recurrence_text: str | None = None
    timezone_text: str | None = None
    all_day_hint: bool | None = None

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator(
        "event_kind",
        "location_text",
        "start_time_text",
        "end_time_text",
        "duration_text",
        "recurrence_text",
        "timezone_text",
    )
    @classmethod
    def optional_strings_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be null or non-empty")
        return value


class CandidateSemanticAssertion(StrictBaseModel):
    canonical_text: str
    memory_type: MemoryType
    subject: EntityMention
    semantic_arguments: list[SemanticArgument]
    predicate_type: str
    modality: Modality
    polarity: Polarity
    certainty: Certainty
    explicitness: Explicitness
    attributed_to: str
    source_spans: list[SourceSpan]
    calendar_event: CalendarEventExtraction | None = None

    @field_validator("canonical_text", "predicate_type", "attributed_to")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_calendar_event_fields(self) -> CandidateSemanticAssertion:
        if self.calendar_event is not None and not self.calendar_event.title.strip():
            raise ValueError("calendar_event.title must not be empty")
        return self


class CompilerOutput(StrictBaseModel):
    assertions: list[CandidateSemanticAssertion]


class ValidationIssue(StrictBaseModel):
    severity: IssueSeverity
    code: str
    message: str
    assertion_index: int | None = None


class ValidationReport(StrictBaseModel):
    valid: bool
    issues: list[ValidationIssue]


class ValidatedAssertionResult(StrictBaseModel):
    assertion: CandidateSemanticAssertion
    validation_report: ValidationReport


class TemporalMetadata(StrictBaseModel):
    temporal_kind: TemporalKind
    valid_from_hint: str | None = None
    valid_to_hint: str | None = None
    event_time: str | None = None
    temporal_precision: TemporalPrecision
    recurrence: Recurrence
    recurrence_specifics: str | None = None
    calendar_event: CalendarEventExtraction | None = None


def _parse_timestamp(value: datetime | str) -> datetime | None:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
