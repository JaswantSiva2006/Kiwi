"""Read-only v1 reconciliation orchestration."""

from __future__ import annotations

import time
from typing import Any, Protocol

from kivi_memory.reconciliation.config import RECONCILIATION_MAX_CANDIDATES
from kivi_memory.reconciliation.formatting import (
    build_compact_input,
    semantic_signature_for_incoming,
    semantic_signature_for_memory,
)
from kivi_memory.reconciliation.judge import OllamaReconciliationJudge
from kivi_memory.reconciliation.models import (
    HydratedMemory,
    JudgeResult,
    ReconciliationDecision,
    ReconciliationFailure,
    ReconciliationOp,
)
from kivi_memory.reconciliation.repository import ReconciliationRepository
from kivi_memory.reconciliation.validator import validate_model_decision


class CandidateRepository(Protocol):
    def hydrate_candidates(self, memory_ids: list[str]) -> list[HydratedMemory]:
        ...


class ReconciliationJudge(Protocol):
    def decide(self, compact_input: str, retry_note: str | None = None) -> JudgeResult:
        ...


def reconcile_memory(
    incoming_assertion: Any,
    final_candidates: list[Any],
    temporal_metadata: Any = None,
    entity_resolution: dict[str, Any] | None = None,
    repository: CandidateRepository | None = None,
    judge: ReconciliationJudge | None = None,
) -> ReconciliationDecision:
    """Decide how one incoming assertion relates to the final reranked candidates.

    This function is read-only. It may hydrate candidate memories and call the
    local judge, but it never writes ledger mutations.
    """

    total_started = time.perf_counter()
    incoming_assertion, temporal_metadata, entity_resolution = _unwrap_incoming(
        incoming_assertion,
        temporal_metadata,
        entity_resolution,
    )
    candidate_ids = _first_ranked_memory_ids(final_candidates)
    if not candidate_ids:
        return _decision(
            op=ReconciliationOp.ADD,
            target_memory_ids=[],
            reason="NO_CANDIDATES_BYPASS",
            total_started=total_started,
        )

    repo = repository or ReconciliationRepository()
    started = time.perf_counter()
    hydrated_candidates = repo.hydrate_candidates(candidate_ids)
    hydration_ms = _elapsed_ms(started)

    started = time.perf_counter()
    compact_input, candidate_map, incoming = build_compact_input(
        incoming_assertion,
        entity_resolution,
        temporal_metadata,
        hydrated_candidates[:RECONCILIATION_MAX_CANDIDATES],
    )
    formatting_ms = _elapsed_ms(started)

    duplicate = _exact_duplicate_target(incoming, hydrated_candidates)
    if duplicate is not None:
        return _decision(
            op=ReconciliationOp.REINFORCE,
            target_memory_ids=[duplicate],
            reason="EXACT_DUPLICATE_BYPASS",
            candidate_map=candidate_map,
            compact_input=compact_input,
            candidate_hydration_ms=hydration_ms,
            formatting_ms=formatting_ms,
            total_started=total_started,
        )

    judge = judge or OllamaReconciliationJudge()
    llm_started = time.perf_counter()
    try:
        judge_result = judge.decide(compact_input)
        llm_ms = judge_result.llm_ms
        validation_started = time.perf_counter()
        try:
            model_decision = validate_model_decision(judge_result.raw_decision, candidate_map)
        except ReconciliationFailure as exc:
            retry_result = judge.decide(compact_input, retry_note=str(exc))
            llm_ms += retry_result.llm_ms
            judge_result = retry_result
            model_decision = validate_model_decision(retry_result.raw_decision, candidate_map)
        validation_ms = _elapsed_ms(validation_started)
    except Exception:
        if "llm_ms" not in locals():
            llm_ms = _elapsed_ms(llm_started)
        raise

    guarded_op = _guarded_operation(
        model_decision.op,
        model_decision.targets,
        candidate_map,
        incoming_assertion,
        temporal_metadata,
        hydrated_candidates,
    )
    return _decision(
        op=guarded_op,
        target_memory_ids=[] if guarded_op == ReconciliationOp.ADD else [candidate_map[target] for target in model_decision.targets],
        reason="LLM_DECISION",
        candidate_map=candidate_map,
        compact_input=compact_input,
        candidate_hydration_ms=hydration_ms,
        formatting_ms=formatting_ms,
        llm_ms=llm_ms,
        validation_ms=validation_ms,
        total_started=total_started,
        prompt_eval_count=judge_result.prompt_eval_count,
        eval_count=judge_result.eval_count,
    )


def _first_ranked_memory_ids(final_candidates: list[Any]) -> list[str]:
    seen = set()
    memory_ids = []
    for candidate in final_candidates:
        memory_id = _get(candidate, "memory_id")
        if memory_id is None:
            continue
        memory_id = str(memory_id).strip()
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        memory_ids.append(memory_id)
        if len(memory_ids) == RECONCILIATION_MAX_CANDIDATES:
            break
    return memory_ids


def _unwrap_incoming(
    incoming_assertion: Any,
    temporal_metadata: Any,
    entity_resolution: dict[str, Any] | None,
) -> tuple[Any, Any, dict[str, Any] | None]:
    if isinstance(incoming_assertion, dict) and "semantic_assertion" in incoming_assertion:
        wrapper = incoming_assertion
        return (
            wrapper["semantic_assertion"],
            temporal_metadata if temporal_metadata is not None else wrapper.get("temporal_metadata"),
            entity_resolution if entity_resolution is not None else wrapper.get("entity_resolution"),
        )
    return incoming_assertion, temporal_metadata, entity_resolution


def _exact_duplicate_target(incoming: Any, hydrated_candidates: list[HydratedMemory]) -> str | None:
    incoming_signature = semantic_signature_for_incoming(incoming)
    for candidate in hydrated_candidates:
        if semantic_signature_for_memory(candidate) == incoming_signature:
            return candidate.memory_id
    return None


def _guarded_operation(
    op: ReconciliationOp,
    target_labels: list[str],
    candidate_map: dict[str, str],
    incoming_assertion: Any,
    temporal_metadata: Any,
    hydrated_candidates: list[HydratedMemory],
) -> ReconciliationOp:
    if op not in {ReconciliationOp.SUPERSEDE, ReconciliationOp.RETRACT} or len(target_labels) != 1:
        return op
    target_id = candidate_map.get(target_labels[0])
    target = next((candidate for candidate in hydrated_candidates if candidate.memory_id == target_id), None)
    if target is None:
        return op
    if _is_bounded_calendar_exception_to_recurring_schedule(incoming_assertion, temporal_metadata, target):
        return ReconciliationOp.ADD
    return op


def _is_bounded_calendar_exception_to_recurring_schedule(
    incoming_assertion: Any,
    temporal_metadata: Any,
    target: HydratedMemory,
) -> bool:
    if str(_get(incoming_assertion, "memory_type") or "").upper() != "CALENDAR_EVENT":
        return False
    if str(target.memory_type).upper() != "CALENDAR_EVENT":
        return False
    target_recurrence = str(_enum_value(target.recurrence) or "").upper()
    incoming_recurrence = str(_enum_value(_get(temporal_metadata, "recurrence")) or "NONE").upper()
    if target_recurrence in {"", "NONE"} or incoming_recurrence not in {"", "NONE"}:
        return False
    text = " ".join(
        part
        for part in (
            _get(incoming_assertion, "canonical_text"),
            " ".join(str(_get(arg, "text") or "") for arg in (_get(incoming_assertion, "semantic_arguments") or [])),
            str(_get(temporal_metadata, "event_time") or ""),
            str(_get(temporal_metadata, "valid_from_hint") or ""),
        )
        if part
    ).casefold()
    bounded_markers = ("this week", "today", "tomorrow", "tonight", "this time", "instead", "on ")
    permanent_markers = ("usually", "normal", "normally", "from now", "going forward", "starting", "permanently", "moved to", "move to", "changed to", "stops", "cancelled", "canceled")
    return any(marker in text for marker in bounded_markers) and not any(marker in text for marker in permanent_markers)


def _decision(
    *,
    op: ReconciliationOp,
    target_memory_ids: list[str],
    reason: str,
    total_started: float,
    candidate_map: dict[str, str] | None = None,
    compact_input: str | None = None,
    candidate_hydration_ms: float = 0.0,
    formatting_ms: float = 0.0,
    llm_ms: float = 0.0,
    validation_ms: float = 0.0,
    prompt_eval_count: int | None = None,
    eval_count: int | None = None,
) -> ReconciliationDecision:
    return ReconciliationDecision(
        op=op,
        target_memory_ids=target_memory_ids,
        reason=reason,
        candidate_map=candidate_map or {},
        compact_input=compact_input,
        candidate_hydration_ms=candidate_hydration_ms,
        formatting_ms=formatting_ms,
        llm_ms=llm_ms,
        validation_ms=validation_ms,
        total_ms=_elapsed_ms(total_started),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
