"""Deterministic validation for candidate semantic assertions."""

from __future__ import annotations

import re

from kivi_memory.common.schemas import (
    CandidateSemanticAssertion,
    CompilerOutput,
    EpisodeMessage,
    MemoryType,
    MemoryEpisode,
    SourceSpan,
    ValidatedAssertionResult,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

_SNAKE_CASE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
_ENTITY_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ -]{0,39}$")
_CALENDAR_EVENT_KINDS = {
    "MEETING",
    "APPOINTMENT",
    "DEADLINE",
    "TASK",
    "TRAVEL",
    "PERSONAL",
    "RECURRING_ACTIVITY",
    "OTHER",
}


def validate_compiler_output(output: CompilerOutput, episode: MemoryEpisode) -> ValidationReport:
    """Return one combined report for a compiler output."""

    issues: list[ValidationIssue] = []
    for result in validate_assertions(output.assertions, episode):
        issues.extend(result.validation_report.issues)
    return _report(issues)


def validate_assertions(
    assertions: list[CandidateSemanticAssertion],
    episode: MemoryEpisode,
) -> list[ValidatedAssertionResult]:
    """Validate assertions without mutating or repairing them."""

    results = [_validate_single_assertion(assertion, episode, index) for index, assertion in enumerate(assertions)]
    canonical_to_indexes: dict[str, list[int]] = {}
    for index, assertion in enumerate(assertions):
        canonical_to_indexes.setdefault(_normalize_text(assertion.canonical_text), []).append(index)

    for indexes in canonical_to_indexes.values():
        if len(indexes) < 2:
            continue
        for index in indexes:
            result = results[index]
            issues = [
                *result.validation_report.issues,
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="DUPLICATE_ASSERTION",
                    message="canonical_text duplicates another assertion after normalization",
                    assertion_index=index,
                ),
            ]
            results[index] = ValidatedAssertionResult(
                assertion=result.assertion,
                validation_report=_report(issues),
            )

    return results


def _validate_single_assertion(
    assertion: CandidateSemanticAssertion,
    episode: MemoryEpisode,
    index: int,
) -> ValidatedAssertionResult:
    issues: list[ValidationIssue] = []
    messages_by_id = {message.message_id: message for message in episode.messages}

    _check_required_assertion_fields(assertion, index, issues)
    _check_source_spans(assertion.source_spans, messages_by_id, index, issues)
    _check_writeback_grounding(assertion.source_spans, messages_by_id, index, issues)
    _check_predicate_type(assertion, index, issues)
    _check_subject_duplication(assertion, index, issues)
    _check_duplicate_arguments(assertion, index, issues)
    _check_duplicate_source_spans(assertion.source_spans, index, issues)
    _check_calendar_event(assertion, index, issues)
    _check_entity_type(assertion.subject.entity_type, "subject.entity_type", index, issues)
    for argument in assertion.semantic_arguments:
        if not argument.is_entity and argument.entity_type is not None:
            issues.append(
                _issue(
                    "ERROR",
                    "INCONSISTENT_ARGUMENT_ENTITY_FIELDS",
                    "semantic argument entity_type must be null when is_entity is false",
                    index,
                )
            )
        if argument.entity_type is not None:
            _check_entity_type(argument.entity_type, "semantic_arguments.entity_type", index, issues)

    return ValidatedAssertionResult(
        assertion=assertion,
        validation_report=_report(issues),
    )


def _check_required_assertion_fields(
    assertion: CandidateSemanticAssertion,
    index: int,
    issues: list[ValidationIssue],
) -> None:
    if not assertion.canonical_text.strip():
        issues.append(_issue("ERROR", "EMPTY_CANONICAL_TEXT", "canonical_text must not be empty", index))
    if not assertion.subject.text.strip():
        issues.append(_issue("ERROR", "EMPTY_SUBJECT_TEXT", "subject text must not be empty", index))
    if not assertion.subject.entity_type.strip():
        issues.append(_issue("ERROR", "EMPTY_SUBJECT_ENTITY_TYPE", "subject entity_type must not be empty", index))
    if not assertion.predicate_type.strip():
        issues.append(_issue("ERROR", "EMPTY_PREDICATE_TYPE", "predicate_type must not be empty", index))
    if not assertion.attributed_to.strip():
        issues.append(_issue("ERROR", "EMPTY_ATTRIBUTED_TO", "attributed_to must not be empty", index))

    for argument in assertion.semantic_arguments:
        if not argument.role.strip():
            issues.append(_issue("ERROR", "EMPTY_ARGUMENT_ROLE", "semantic argument role must not be empty", index))
        if not argument.text.strip():
            issues.append(_issue("ERROR", "EMPTY_ARGUMENT_TEXT", "semantic argument text must not be empty", index))


def _check_source_spans(
    source_spans: list[SourceSpan],
    messages_by_id: dict[str, EpisodeMessage],
    index: int,
    issues: list[ValidationIssue],
) -> None:
    if not source_spans:
        issues.append(_issue("ERROR", "MISSING_SOURCE_SPAN", "assertion must contain at least one source span", index))
        return

    for span in source_spans:
        if not span.message_id.strip():
            issues.append(_issue("ERROR", "EMPTY_SOURCE_MESSAGE_ID", "source span message_id must not be empty", index))
        if not span.text.strip():
            issues.append(_issue("ERROR", "EMPTY_SOURCE_TEXT", "source span text must not be empty", index))

        message = messages_by_id.get(span.message_id)
        if message is None:
            issues.append(
                _issue(
                    "ERROR",
                    "SOURCE_MESSAGE_NOT_FOUND",
                    f"source span references unknown message_id {span.message_id!r}",
                    index,
                )
            )
            continue

        if span.text not in message.text:
            issues.append(
                _issue(
                    "ERROR",
                    "SOURCE_SPAN_NOT_VERBATIM",
                    f"source span text does not occur verbatim in message {span.message_id!r}",
                    index,
                )
            )


def _check_writeback_grounding(
    source_spans: list[SourceSpan],
    messages_by_id: dict[str, EpisodeMessage],
    index: int,
    issues: list[ValidationIssue],
) -> None:
    if not any(message.memory_eligible is not None or message.context_only is not None for message in messages_by_id.values()):
        return
    for span in source_spans:
        message = messages_by_id.get(span.message_id)
        if (
            message
            and message.role.value == "USER"
            and message.memory_eligible is True
            and message.context_only is not True
        ):
            return
    issues.append(
        _issue(
            "ERROR",
            "MISSING_MEMORY_ELIGIBLE_USER_GROUNDING",
            "writeback assertion must be grounded in at least one eligible target USER message",
            index,
        )
    )


def _check_predicate_type(
    assertion: CandidateSemanticAssertion,
    index: int,
    issues: list[ValidationIssue],
) -> None:
    if assertion.predicate_type.strip() and not _SNAKE_CASE_RE.fullmatch(assertion.predicate_type):
        issues.append(
            _issue(
                "ERROR",
                "MALFORMED_PREDICATE_TYPE",
                "predicate_type must use uppercase snake case",
                index,
            )
        )


def _check_subject_duplication(
    assertion: CandidateSemanticAssertion,
    index: int,
    issues: list[ValidationIssue],
) -> None:
    subject = _normalize_text(assertion.subject.text)
    for argument in assertion.semantic_arguments:
        if _normalize_text(argument.text) == subject:
            issues.append(
                _issue(
                    "WARNING",
                    "SUBJECT_DUPLICATED_IN_ARGUMENTS",
                    "semantic argument repeats the subject text",
                    index,
                )
            )


def _check_duplicate_arguments(
    assertion: CandidateSemanticAssertion,
    index: int,
    issues: list[ValidationIssue],
) -> None:
    seen: set[tuple[str, str]] = set()
    for argument in assertion.semantic_arguments:
        key = (_normalize_text(argument.role), _normalize_text(argument.text))
        if key in seen:
            issues.append(_issue("WARNING", "DUPLICATE_ARGUMENT", "duplicate semantic argument", index))
        seen.add(key)


def _check_duplicate_source_spans(
    source_spans: list[SourceSpan],
    index: int,
    issues: list[ValidationIssue],
) -> None:
    seen: set[tuple[str, str]] = set()
    for span in source_spans:
        key = (span.message_id, span.text)
        if key in seen:
            issues.append(_issue("WARNING", "DUPLICATE_SOURCE_SPAN", "duplicate source span", index))
        seen.add(key)


def _check_calendar_event(
    assertion: CandidateSemanticAssertion,
    index: int,
    issues: list[ValidationIssue],
) -> None:
    if assertion.calendar_event is None:
        return
    calendar_event = assertion.calendar_event
    if not calendar_event.title.strip():
        issues.append(_issue("ERROR", "EMPTY_CALENDAR_EVENT_TITLE", "calendar_event.title must not be empty", index))
    if not (calendar_event.start_time_text or calendar_event.recurrence_text):
        issues.append(
            _issue(
                "ERROR",
                "MISSING_CALENDAR_SCHEDULE_SIGNAL",
                "CALENDAR_EVENT requires start_time_text or recurrence_text",
                index,
            )
        )
    if calendar_event.event_kind is not None and calendar_event.event_kind not in _CALENDAR_EVENT_KINDS:
        issues.append(
            _issue(
                "WARNING",
                "UNKNOWN_CALENDAR_EVENT_KIND",
                "calendar_event.event_kind is outside the recommended lightweight vocabulary",
                index,
            )
        )


def _check_entity_type(value: str, field_name: str, index: int, issues: list[ValidationIssue]) -> None:
    punctuation_count = sum(1 for character in value if not character.isalnum() and character not in {"_", "-", " "})
    if "." in value or punctuation_count > 1 or not _ENTITY_TYPE_RE.fullmatch(value):
        issues.append(
            _issue(
                "WARNING",
                "MALFORMED_ENTITY_TYPE",
                f"{field_name} looks sentence-like or punctuation-heavy",
                index,
            )
        )


def _issue(severity: str, code: str, message: str, assertion_index: int) -> ValidationIssue:
    return ValidationIssue(
        severity=ValidationSeverity(severity),
        code=code,
        message=message,
        assertion_index=assertion_index,
    )


def _report(issues: list[ValidationIssue]) -> ValidationReport:
    return ValidationReport(
        valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
        issues=issues,
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())
