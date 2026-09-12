"""Temporal extraction and normalization through local Qwen."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from pydantic import ValidationError
from pydantic import BaseModel, ConfigDict

from kivi_memory.common.config import KiviCompilerConfig, load_config_from_env
from kivi_memory.common.schemas import (
    CalendarEventExtraction,
    CandidateSemanticAssertion,
    MemoryType,
    MemoryEpisode,
    Recurrence,
    TemporalMetadata,
    TemporalKind,
    TemporalPrecision,
)
from kivi_memory.enrichment.temporal_prompt import (
    TEMPORAL_NORMALIZATION_SYSTEM_PROMPT,
    TEMPORAL_RETRY_INSTRUCTION,
    TEMPORAL_SYSTEM_PROMPT,
)
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError

logger = logging.getLogger(__name__)

_ISO_DATE_OR_DATETIME_RE = re.compile(
    r"^\d{4}(-\d{2}){0,2}(T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)?)?$"
)
_ISO_INTERVAL_RE = re.compile(r"^.+/.+$")
_RECURRENCE_SPEC_MAX_WORDS = 5
_RECURRENCE_SPEC_MAX_CHARS = 48


class TemporalNormalizerError(RuntimeError):
    """Raised when temporal normalization cannot produce valid metadata."""


@dataclass(frozen=True)
class TemporalResult:
    assertion: CandidateSemanticAssertion
    temporal_metadata: TemporalMetadata
    model_name: str
    inference_duration_ms: float | None
    reasoning_metadata: "TemporalReasoningOutput | None" = None
    normalization_metadata: "TemporalMetadata | None" = None
    reasoning_duration_ms: float = 0.0
    normalization_duration_ms: float = 0.0


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalReasoningOutput(_StrictBaseModel):
    temporal_kind: TemporalKind
    temporal_precision: TemporalPrecision
    recurrence: Recurrence
    recurrence_specifics: str | None = None


class TemporalNormalizationOutput(_StrictBaseModel):
    valid_from_hint: str | None = None
    valid_to_hint: str | None = None
    event_time: str | None = None
    calendar_event: CalendarEventExtraction | None = None


class TemporalNormalizer:
    def __init__(
        self,
        client: OllamaClient | None = None,
        config: KiviCompilerConfig | None = None,
    ) -> None:
        self.config = config or load_config_from_env()
        self.client = client or OllamaClient(
            base_url=self.config.ollama_base_url,
            timeout_seconds=self.config.timeout_seconds,
        )

    def normalize(self, assertion: CandidateSemanticAssertion, episode: MemoryEpisode) -> TemporalResult:
        return self.normalize_both(assertion, episode)

    def normalize_none(self, assertion: CandidateSemanticAssertion) -> TemporalResult:
        metadata = none_temporal_metadata()
        return TemporalResult(
            assertion=assertion,
            temporal_metadata=metadata,
            model_name="SKIPPED",
            inference_duration_ms=0.0,
            reasoning_metadata=None,
            normalization_metadata=None,
            reasoning_duration_ms=0.0,
            normalization_duration_ms=0.0,
        )

    def normalize_reason_only(self, assertion: CandidateSemanticAssertion, episode: MemoryEpisode) -> TemporalResult:
        started = time.perf_counter()
        reasoning_started = time.perf_counter()
        reasoning = self._run_reasoning_pass(assertion, episode)
        reasoning_ms = (time.perf_counter() - reasoning_started) * 1000
        normalization = TemporalNormalizationOutput()
        metadata = _merge_temporal_metadata(assertion, reasoning, normalization)
        _validate_temporal_metadata(metadata)
        return TemporalResult(
            assertion=assertion,
            temporal_metadata=metadata,
            model_name=self.config.temporal_reasoning_model,
            inference_duration_ms=(time.perf_counter() - started) * 1000,
            reasoning_metadata=reasoning,
            normalization_metadata=None,
            reasoning_duration_ms=reasoning_ms,
            normalization_duration_ms=0.0,
        )

    def normalize_both(self, assertion: CandidateSemanticAssertion, episode: MemoryEpisode) -> TemporalResult:
        started = time.perf_counter()
        reasoning_started = time.perf_counter()
        reasoning = self._run_reasoning_pass(assertion, episode)
        reasoning_ms = (time.perf_counter() - reasoning_started) * 1000
        normalization_started = time.perf_counter()
        normalization = self._run_normalization_pass(assertion, episode, reasoning)
        normalization_ms = (time.perf_counter() - normalization_started) * 1000
        metadata = _merge_temporal_metadata(assertion, reasoning, normalization)
        _validate_temporal_metadata(metadata)
        return TemporalResult(
            assertion=assertion,
            temporal_metadata=metadata,
            model_name=f"{self.config.temporal_reasoning_model}+{self.config.temporal_normalization_model}",
            inference_duration_ms=(time.perf_counter() - started) * 1000,
            reasoning_metadata=reasoning,
            normalization_metadata=normalization,
            reasoning_duration_ms=reasoning_ms,
            normalization_duration_ms=normalization_ms,
        )

    def _run_reasoning_pass(
        self,
        assertion: CandidateSemanticAssertion,
        episode: MemoryEpisode,
    ) -> TemporalReasoningOutput:
        return self._run_structured_temporal_call(
            model=self.config.temporal_reasoning_model,
            system_prompt=TEMPORAL_SYSTEM_PROMPT,
            user_content=format_temporal_input(assertion, episode),
            output_model=TemporalReasoningOutput,
            think=False,
            label="temporal reasoning",
            episode=episode,
        )

    def _run_normalization_pass(
        self,
        assertion: CandidateSemanticAssertion,
        episode: MemoryEpisode,
        reasoning: TemporalReasoningOutput,
    ) -> TemporalNormalizationOutput:
        return self._run_structured_temporal_call(
            model=self.config.temporal_normalization_model,
            system_prompt=TEMPORAL_NORMALIZATION_SYSTEM_PROMPT,
            user_content=format_temporal_normalization_input(assertion, episode, reasoning),
            output_model=TemporalNormalizationOutput,
            think=False,
            label="temporal value normalization",
            episode=episode,
            metadata_validator=_validate_temporal_normalization_output,
        )

    def _run_structured_temporal_call(
        self,
        *,
        model,
        system_prompt,
        user_content,
        output_model,
        think,
        label,
        episode,
        metadata_validator=None,
    ):
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            content = user_content if attempt == 1 else f"{user_content}\n\n{TEMPORAL_RETRY_INSTRUCTION}"
            if attempt > 1:
                logger.warning("Retrying %s episode=%s attempt=%s", label, episode.episode_id, attempt)
            try:
                raw = self.client.chat_structured(
                    model=model,
                    system_prompt=system_prompt,
                    user_content=content,
                    json_schema=output_model.model_json_schema(),
                    temperature=0,
                    think=think,
                )
                output = output_model.model_validate(raw)
                if metadata_validator is not None:
                    metadata_validator(output)
                return output
            except (OllamaClientError, ValidationError, ValueError) as exc:
                last_error = exc
        raise TemporalNormalizerError(f"{label} failed after {self.config.max_attempts} attempt(s): {last_error}")


def format_temporal_input(assertion: CandidateSemanticAssertion, episode: MemoryEpisode) -> str:
    messages_by_id = {message.message_id: message for message in episode.messages}
    lines = [
        "<assertion>",
        assertion.canonical_text,
        "</assertion>",
        "",
        "<assertion_metadata>",
        f"memory_type={assertion.memory_type}",
        f"predicate_type={assertion.predicate_type}",
        "</assertion_metadata>",
        "",
    ]

    for span in assertion.source_spans:
        message = messages_by_id.get(span.message_id)
        timestamp = message.timestamp if message is not None else "UNKNOWN"
        lines.extend(
            [
                f'<source_text message_id="{span.message_id}">',
                span.text,
                "</source_text>",
                "",
                f'<reference_timestamp message_id="{span.message_id}">',
                str(timestamp),
                "</reference_timestamp>",
                "",
            ]
        )

    lines.extend(
        [
            "<reference_timezone>",
            episode.timezone or "",
            "</reference_timezone>",
            "",
            "<locale>",
            episode.locale or "",
            "</locale>",
        ]
    )
    return "\n".join(lines)


def format_temporal_normalization_input(
    assertion: CandidateSemanticAssertion,
    episode: MemoryEpisode,
    reasoning: TemporalReasoningOutput,
) -> str:
    return "\n".join(
        [
            format_temporal_input(assertion, episode),
            "",
            "<decided_temporal_structure>",
            f"temporal_kind={reasoning.temporal_kind}",
            f"temporal_precision={reasoning.temporal_precision}",
            f"recurrence={reasoning.recurrence}",
            f"recurrence_specifics={reasoning.recurrence_specifics or ''}",
            "</decided_temporal_structure>",
        ]
    )


def _merge_temporal_metadata(
    assertion: CandidateSemanticAssertion,
    reasoning: TemporalReasoningOutput,
    normalization: TemporalNormalizationOutput,
) -> TemporalMetadata:
    valid_from_hint = normalization.valid_from_hint
    valid_to_hint = normalization.valid_to_hint
    event_time = normalization.event_time
    temporal_kind = reasoning.temporal_kind
    if reasoning.recurrence != Recurrence.NONE:
        temporal_kind = TemporalKind.RECURRENCE

    if temporal_kind == TemporalKind.NONE:
        valid_from_hint = None
        valid_to_hint = None
        event_time = None
        recurrence = Recurrence.NONE
        recurrence_specifics = None
    elif temporal_kind == TemporalKind.STATE_INTERVAL:
        event_time = None
        recurrence = Recurrence.NONE
        recurrence_specifics = None
    elif temporal_kind == TemporalKind.RECURRENCE:
        event_time = None
        recurrence = reasoning.recurrence
        recurrence_specifics = _sanitize_recurrence_specifics(reasoning.recurrence_specifics)
        if valid_from_hint is None:
            valid_from_hint = _valid_from_from_recurrence_specifics(recurrence_specifics)
        if valid_to_hint is None:
            valid_to_hint = _valid_to_from_recurrence_specifics(recurrence_specifics)
    else:
        valid_from_hint = None
        valid_to_hint = None
        recurrence = Recurrence.NONE
        recurrence_specifics = None

    return TemporalMetadata(
        temporal_kind=temporal_kind,
        valid_from_hint=valid_from_hint,
        valid_to_hint=valid_to_hint,
        event_time=event_time,
        temporal_precision=reasoning.temporal_precision,
        recurrence=recurrence,
        recurrence_specifics=recurrence_specifics,
        calendar_event=_calendar_event_payload(assertion, normalization),
    )


def _calendar_event_payload(
    assertion: CandidateSemanticAssertion,
    normalization: TemporalNormalizationOutput,
) -> CalendarEventExtraction | None:
    if assertion.memory_type != MemoryType.CALENDAR_EVENT:
        return None
    if normalization.calendar_event is not None:
        return normalization.calendar_event
    return _derive_calendar_event(assertion)


def _derive_calendar_event(assertion: CandidateSemanticAssertion) -> CalendarEventExtraction:
    text = assertion.canonical_text
    title = _derive_calendar_title(text)
    schedule_parts = [
        argument.text
        for argument in assertion.semantic_arguments
        if not argument.is_entity and _looks_calendar_argument(argument.role, argument.text)
    ]
    schedule_text = " ".join(schedule_parts) or text
    return CalendarEventExtraction(
        title=title,
        event_kind=_derive_event_kind(text),
        location_text=_argument_text(assertion, {"location", "place", "venue"}),
        start_time_text=schedule_text,
        end_time_text=None,
        duration_text=_argument_text(assertion, {"duration"}),
        recurrence_text=schedule_text if _has_recurrence_text(schedule_text) else None,
        timezone_text=None,
        all_day_hint=False if _has_clock_time(schedule_text) else None,
    )


def _derive_calendar_title(text: str) -> str:
    match = re.search(r"\b(?:have|has|had)\s+(?:a|an|the)?\s*([^.,;]+)", text, flags=re.IGNORECASE)
    if match:
        title = match.group(1).strip()
    else:
        title = text.strip().rstrip(".")
    title = re.sub(r"\b(every|on|at|starting|from|until|next|this)\b.*$", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"\bwith\s+([a-z])", lambda item: "with " + item.group(1).upper(), title, flags=re.IGNORECASE)
    return title[:1].upper() + title[1:] if title else "Calendar event"


def _derive_event_kind(text: str) -> str:
    lowered = text.casefold()
    if "review" in lowered or "meeting" in lowered or "sync" in lowered:
        return "MEETING"
    if "deadline" in lowered or "due" in lowered:
        return "DEADLINE"
    if "appointment" in lowered:
        return "APPOINTMENT"
    return "OTHER"


def _argument_text(assertion: CandidateSemanticAssertion, roles: set[str]) -> str | None:
    for argument in assertion.semantic_arguments:
        if argument.role.casefold() in roles and argument.text.strip():
            return argument.text
    return None


def _looks_calendar_argument(role: str, text: str) -> bool:
    value = f"{role} {text}".casefold()
    return any(token in value for token in ("time", "frequency", "recurrence", "every", "daily", "weekly", "monthly", "yearly", "starting"))


def _has_recurrence_text(text: str) -> bool:
    return any(token in text.casefold() for token in ("every", "daily", "weekly", "monthly", "yearly"))


def _has_clock_time(text: str) -> bool:
    return re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", text, flags=re.IGNORECASE) is not None


def _sanitize_recurrence_specifics(value: str | None) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.strip().split())
    if not compact:
        return None
    if len(compact) <= _RECURRENCE_SPEC_MAX_CHARS and len(compact.split()) <= _RECURRENCE_SPEC_MAX_WORDS:
        return compact
    extracted = _extract_compact_recurrence_specifics(compact)
    return extracted


def _extract_compact_recurrence_specifics(value: str) -> str | None:
    text = value.casefold()
    weekday_match = re.search(
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
        flags=re.IGNORECASE,
    )
    daypart_match = re.search(r"\b(morning|afternoon|evening|night)\b", text, flags=re.IGNORECASE)
    time_match = re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", text, flags=re.IGNORECASE)
    twenty_four_hour_match = re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)?\b", text, flags=re.IGNORECASE)
    if weekday_match:
        parts = [weekday_match.group(1).upper()]
        if time_match:
            parts.append(time_match.group(0).upper())
        elif daypart_match:
            parts.append(daypart_match.group(1).upper())
        elif twenty_four_hour_match:
            parts.append(_compact_time(twenty_four_hour_match.group(0)))
        return " ".join(parts)

    month_day_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",
        text,
        flags=re.IGNORECASE,
    )
    if month_day_match:
        return month_day_match.group(0).upper()

    day_of_month_match = re.search(r"\bday\s+\d{1,2}\b", text, flags=re.IGNORECASE)
    if day_of_month_match:
        return day_of_month_match.group(0).upper()

    return None


def _compact_time(value: str) -> str:
    return re.sub(r"(?::00)?(?:[+-]\d{2}:\d{2}|Z)?$", "", value)


def _valid_from_from_recurrence_specifics(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"STARTING[^,;]*_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)?)", value)
    return match.group(1) if match else None


def _valid_to_from_recurrence_specifics(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\bUNTIL\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)?)", value)
    return match.group(1) if match else None


def none_temporal_metadata() -> TemporalMetadata:
    return TemporalMetadata(
        temporal_kind=TemporalKind.NONE,
        valid_from_hint=None,
        valid_to_hint=None,
        event_time=None,
        temporal_precision=TemporalPrecision.NONE,
        recurrence=Recurrence.NONE,
        recurrence_specifics=None,
    )


def _validate_temporal_metadata(metadata: TemporalMetadata) -> None:
    values = [metadata.valid_from_hint, metadata.valid_to_hint, metadata.event_time]
    for value in values:
        if value is not None and not _looks_like_iso_temporal(value):
            raise ValueError(f"temporal value is not ISO-like: {value!r}")

    if metadata.valid_from_hint and metadata.valid_to_hint:
        if _comparable(metadata.valid_from_hint, metadata.valid_to_hint):
            if metadata.valid_from_hint > metadata.valid_to_hint:
                raise ValueError("valid_from_hint must be <= valid_to_hint")

    if metadata.recurrence_specifics is not None and not metadata.recurrence_specifics.strip():
        raise ValueError("recurrence_specifics must not be blank when present")

    # NONE with a temporal value is inconsistent, but the contract describes
    # this as a "should normally" check rather than a structural failure.


def _validate_temporal_normalization_output(metadata: TemporalNormalizationOutput) -> None:
    values = [metadata.valid_from_hint, metadata.valid_to_hint, metadata.event_time]
    for value in values:
        if value is not None and not _looks_like_iso_temporal(value):
            raise ValueError(f"temporal value is not ISO-like: {value!r}")

    if metadata.valid_from_hint and metadata.valid_to_hint:
        if _comparable(metadata.valid_from_hint, metadata.valid_to_hint):
            if metadata.valid_from_hint > metadata.valid_to_hint:
                raise ValueError("valid_from_hint must be <= valid_to_hint")


def _looks_like_iso_temporal(value: str) -> bool:
    if _ISO_INTERVAL_RE.fullmatch(value):
        start, end = value.split("/", 1)
        return _looks_like_iso_temporal(start) and _looks_like_iso_temporal(end)
    return _ISO_DATE_OR_DATETIME_RE.fullmatch(value) is not None


def _comparable(left: str, right: str) -> bool:
    return "/" not in left and "/" not in right and len(left) == len(right)
