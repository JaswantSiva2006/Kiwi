from __future__ import annotations

import re
from uuid import uuid4

import pytest

from kivi_memory.reconciliation import ReconciliationFailure, ReconciliationOp, reconcile_memory
from kivi_memory.reconciliation.models import HydratedMemory, HydratedMemoryArgument, JudgeResult


def assertion(
    canonical_text: str,
    *,
    subject_entity_id: str | None = None,
    predicate_type: str = "RELATED_TO",
    argument_entities: list[tuple[str, str]] | None = None,
    polarity: str = "POSITIVE",
    modality: str = "FACT",
    memory_type: str = "ENTITY_CONTEXT",
):
    entity_resolution_args = []
    semantic_arguments = []
    for role, entity_id in argument_entities or []:
        semantic_arguments.append({"role": role, "text": role, "is_entity": True, "entity_type": "entity"})
        entity_resolution_args.append({"role": role, "result": {"resolution": "MATCHED", "entity_id": entity_id}})
    return {
        "semantic_assertion": {
            "canonical_text": canonical_text,
            "memory_type": memory_type,
            "subject": {"text": "subject", "entity_type": "person"},
            "semantic_arguments": semantic_arguments,
            "predicate_type": predicate_type,
            "modality": modality,
            "polarity": polarity,
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "the user",
            "source_spans": [{"message_id": "m1", "text": canonical_text}],
        },
        "temporal_metadata": {
            "temporal_kind": "NONE",
            "valid_from_hint": None,
            "valid_to_hint": None,
            "event_time": None,
            "temporal_precision": "NONE",
            "recurrence": "NONE",
            "recurrence_specifics": None,
        },
        "entity_resolution": {
            "subject": {"resolution": "MATCHED", "entity_id": subject_entity_id},
            "semantic_arguments": entity_resolution_args,
        },
    }


def memory(
    memory_id: str,
    canonical_text: str,
    *,
    subject_entity_id: str | None = None,
    predicate_type: str = "RELATED_TO",
    argument_entities: list[tuple[str, str]] | None = None,
    polarity: str = "POSITIVE",
    modality: str = "FACT",
    memory_type: str = "ENTITY_CONTEXT",
    temporal_kind: str = "NONE",
    recurrence: str = "NONE",
    recurrence_specifics: str | None = None,
) -> HydratedMemory:
    return HydratedMemory(
        memory_id=memory_id,
        canonical_text=canonical_text,
        subject_entity_id=subject_entity_id,
        subject_text="subject",
        predicate_type=predicate_type,
        memory_type=memory_type,
        modality=modality,
        polarity=polarity,
        certainty="CERTAIN",
        temporal_kind=temporal_kind,
        valid_from=None,
        valid_to=None,
        event_time=None,
        temporal_precision="NONE",
        recurrence=recurrence,
        recurrence_specifics=recurrence_specifics,
        status="ACTIVE",
        arguments=[
            HydratedMemoryArgument(role=role, text=role, is_entity=True, entity_id=entity_id, entity_type="entity", position=index)
            for index, (role, entity_id) in enumerate(argument_entities or [])
        ],
    )


class FakeRepository:
    def __init__(self, memories: list[HydratedMemory]) -> None:
        self.memories = {stored.memory_id: stored for stored in memories}
        self.calls = 0

    def hydrate_candidates(self, memory_ids: list[str]) -> list[HydratedMemory]:
        self.calls += 1
        return [self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories]


class FakeJudge:
    def __init__(self, *decisions: dict) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, compact_input: str, retry_note: str | None = None) -> JudgeResult:
        self.calls += 1
        decision = self.decisions[min(self.calls - 1, len(self.decisions) - 1)]
        return JudgeResult(raw_decision=decision, llm_ms=1.0, prompt_eval_count=42, eval_count=8)


def candidate(memory_id: str) -> dict:
    return {"memory_id": memory_id, "canonical_text": "candidate"}


def test_no_candidates_add_without_llm_call() -> None:
    judge = FakeJudge({"op": "NO_MEMORY", "targets": []})

    result = reconcile_memory(assertion("Priya handles Atlas."), [], repository=FakeRepository([]), judge=judge)

    assert result.op == ReconciliationOp.ADD
    assert result.target_memory_ids == []
    assert result.reason == "NO_CANDIDATES_BYPASS"
    assert judge.calls == 0


def test_exact_duplicate_reinforce_without_llm_call() -> None:
    subject_id = str(uuid4())
    atlas_id = str(uuid4())
    memory_id = str(uuid4())
    repo = FakeRepository([
        memory(
            memory_id,
            "Priya handles Atlas!",
            subject_entity_id=subject_id,
            predicate_type="HANDLES",
            argument_entities=[("project", atlas_id)],
        )
    ])
    judge = FakeJudge({"op": "ADD", "targets": []})

    result = reconcile_memory(
        assertion(
            "Priya handles Atlas.",
            subject_entity_id=subject_id,
            predicate_type="HANDLES",
            argument_entities=[("project", atlas_id)],
        ),
        [candidate(memory_id)],
        repository=repo,
        judge=judge,
    )

    assert result.op == ReconciliationOp.REINFORCE
    assert result.target_memory_ids == [memory_id]
    assert result.reason == "EXACT_DUPLICATE_BYPASS"
    assert judge.calls == 0


def test_exact_duplicate_with_stored_utc_temporal_interval_reinforces_without_llm() -> None:
    subject_id = str(uuid4())
    memory_id = str(uuid4())
    stored = memory(
        memory_id,
        "The user worked on Orion from January 2024 until March 2025.",
        subject_entity_id=subject_id,
        predicate_type="WORKED_ON",
    )
    stored = type(stored)(
        **{
            **stored.__dict__,
            "temporal_kind": "STATE_INTERVAL",
            "valid_from": "2023-12-31T18:30:00+00:00",
            "valid_to": "2025-03-31T18:29:59+00:00",
            "temporal_precision": "MONTH",
        }
    )
    incoming = assertion(
        "The user worked on Orion from January 2024 until March 2025.",
        subject_entity_id=subject_id,
        predicate_type="WORKED_ON",
    )
    incoming["temporal_metadata"] = {
        "temporal_kind": "STATE_INTERVAL",
        "valid_from_hint": "2024-01-01T00:00:00+05:30",
        "valid_to_hint": "2025-03-31T23:59:59+05:30",
        "event_time": None,
        "temporal_precision": "MONTH",
        "recurrence": "NONE",
        "recurrence_specifics": None,
    }
    judge = FakeJudge({"op": "SUPERSEDE", "targets": ["C1"]})

    result = reconcile_memory(incoming, [candidate(memory_id)], repository=FakeRepository([stored]), judge=judge)

    assert result.op == ReconciliationOp.REINFORCE
    assert judge.calls == 0


@pytest.mark.parametrize(
    ("old_text", "new_text", "op"),
    [
        ("Priya handles Atlas.", "Priya is responsible for Atlas.", "REINFORCE"),
        ("Rohit works on Atlas.", "Rohit moved to Payments.", "SUPERSEDE"),
        ("User likes black coffee.", "That earlier statement was wrong; the user never liked black coffee.", "RETRACT"),
        ("User likes black coffee.", "User no longer likes black coffee.", "SUPERSEDE"),
        ("User plays chess.", "User plays badminton.", "ADD"),
    ],
)
def test_non_obvious_cases_go_through_judge(old_text: str, new_text: str, op: str) -> None:
    subject_id = str(uuid4())
    memory_id = str(uuid4())
    repo = FakeRepository([memory(memory_id, old_text, subject_entity_id=subject_id, predicate_type="LIKES")])
    judge = FakeJudge({"op": op, "targets": ["C1"] if op != "ADD" else []})

    result = reconcile_memory(
        assertion(new_text, subject_entity_id=subject_id, predicate_type="LIKES"),
        [candidate(memory_id)],
        repository=repo,
        judge=judge,
    )

    assert result.op == ReconciliationOp(op)
    assert judge.calls == 1
    if op == "ADD":
        assert result.target_memory_ids == []
    else:
        assert result.target_memory_ids == [memory_id]


def test_one_week_calendar_exception_does_not_supersede_recurring_schedule() -> None:
    payments_id = str(uuid4())
    normal_schedule_id = str(uuid4())
    incoming = assertion(
        "The Payments review is scheduled for Thursday at 11 AM this week.",
        subject_entity_id=payments_id,
        predicate_type="SCHEDULED",
        memory_type="CALENDAR_EVENT",
    )
    incoming["temporal_metadata"] = {
        "temporal_kind": "DISCRETE_EVENT",
        "valid_from_hint": None,
        "valid_to_hint": None,
        "event_time": "2026-04-09T11:00:00+05:30",
        "temporal_precision": "EXACT",
        "recurrence": "NONE",
        "recurrence_specifics": None,
    }
    repo = FakeRepository([
        memory(
            normal_schedule_id,
            "The Payments review is usually Wednesday at 2 PM.",
            subject_entity_id=payments_id,
            predicate_type="RECURRING_SCHEDULE",
            memory_type="CALENDAR_EVENT",
            temporal_kind="RECURRENCE",
            recurrence="WEEKLY",
            recurrence_specifics="WEDNESDAY 2 PM",
        )
    ])
    judge = FakeJudge({"op": "SUPERSEDE", "targets": ["C1"]})

    result = reconcile_memory(incoming, [candidate(normal_schedule_id)], repository=repo, judge=judge)

    assert result.op == ReconciliationOp.ADD
    assert result.target_memory_ids == []
    assert judge.calls == 1


def test_model_returns_candidate_id_not_supplied_fails_after_retry() -> None:
    memory_id = str(uuid4())
    repo = FakeRepository([memory(memory_id, "Old fact.")])
    judge = FakeJudge({"op": "REINFORCE", "targets": ["C9"]})

    with pytest.raises(ReconciliationFailure, match="invalid candidate"):
        reconcile_memory(assertion("New fact."), [candidate(memory_id)], repository=repo, judge=judge)

    assert judge.calls == 2


def test_incorrect_target_cardinality_fails_after_retry() -> None:
    memory_id = str(uuid4())
    repo = FakeRepository([memory(memory_id, "Old fact.")])
    judge = FakeJudge({"op": "REINFORCE", "targets": []})

    with pytest.raises(ReconciliationFailure, match="requires exactly one target"):
        reconcile_memory(assertion("New fact."), [candidate(memory_id)], repository=repo, judge=judge)

    assert judge.calls == 2


def test_entity_token_mapping_reuses_same_uuid() -> None:
    entity_id = str(uuid4())
    memory_id = str(uuid4())
    repo = FakeRepository([memory(memory_id, "Subject knows target.", subject_entity_id=entity_id, argument_entities=[("target", entity_id)])])
    judge = FakeJudge({"op": "ADD", "targets": []})

    result = reconcile_memory(
        assertion("Subject knows target.", subject_entity_id=entity_id, argument_entities=[("target", entity_id)]),
        [candidate(memory_id)],
        repository=repo,
        judge=judge,
    )

    assert "s=E1" in result.compact_input
    assert "target:E1" in result.compact_input


def test_final_input_contains_no_full_uuids() -> None:
    entity_id = str(uuid4())
    memory_id = str(uuid4())
    repo = FakeRepository([memory(memory_id, "Old fact.", subject_entity_id=entity_id)])
    judge = FakeJudge({"op": "ADD", "targets": []})

    result = reconcile_memory(assertion("New fact.", subject_entity_id=entity_id), [candidate(memory_id)], repository=repo, judge=judge)

    assert memory_id not in result.compact_input
    assert entity_id not in result.compact_input
    assert re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", result.compact_input) is None


def test_only_one_llm_call_for_non_bypass_reconciliation() -> None:
    memory_id = str(uuid4())
    repo = FakeRepository([memory(memory_id, "Old fact.")])
    judge = FakeJudge({"op": "ADD", "targets": []})

    reconcile_memory(assertion("New fact."), [candidate(memory_id)], repository=repo, judge=judge)

    assert judge.calls == 1


def test_more_than_8_candidates_use_first_8_and_preserve_ranking() -> None:
    memory_ids = [str(uuid4()) for _ in range(10)]
    repo = FakeRepository([memory(memory_id, f"Memory {index}") for index, memory_id in enumerate(memory_ids)])
    judge = FakeJudge({"op": "REINFORCE", "targets": ["C8"]})

    result = reconcile_memory(assertion("New fact."), [candidate(memory_id) for memory_id in memory_ids], repository=repo, judge=judge)

    assert list(result.candidate_map.values()) == memory_ids[:8]
    assert result.target_memory_ids == [memory_ids[7]]
    assert memory_ids[8] not in result.candidate_map.values()
