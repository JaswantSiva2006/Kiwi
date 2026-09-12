from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from kivi_memory.common.schemas import CandidateSemanticAssertion, CompilerOutput, EpisodeMessage, MemoryEpisode
from kivi_memory.enrichment.temporal import TemporalResult, none_temporal_metadata
from kivi_memory.enrichment.temporal_gate import TemporalGateResult, TemporalRoute
from kivi_memory.ledger.mutations import LedgerMutationResult
from kivi_memory.pipeline.memory_pipeline import MemoryPipeline
from kivi_memory.reconciliation import ReconciliationDecision, ReconciliationOp


class FakeCompiler:
    def __init__(self, assertions):
        self.assertions = assertions

    def compile(self, episode):
        del episode
        return SimpleNamespace(
            output=CompilerOutput(assertions=self.assertions),
            model_name="compiler-test",
            inference_duration_ms=1,
            attempt_count=1,
        )


class FakeGate:
    def __init__(self):
        self.calls = 0
        self.last_assertions = None

    def route(self, assertions):
        self.calls += 1
        self.last_assertions = assertions
        return TemporalGateResult(
            routes={"A1": TemporalRoute.NONE, "A2": TemporalRoute.REASON, "A3": TemporalRoute.BOTH},
            temporal_gate_ms=7.0,
            gate_prompt_eval_count=11,
            gate_eval_count=6,
            gate_model_call_count=1,
        )


class FakeTemporalNormalizer:
    def __init__(self):
        self.none_calls = 0
        self.reason_calls = 0
        self.both_calls = 0

    def normalize_none(self, assertion):
        self.none_calls += 1
        return TemporalResult(assertion, none_temporal_metadata(), "SKIPPED", 0.0)

    def normalize_reason_only(self, assertion, episode):
        del episode
        self.reason_calls += 1
        return TemporalResult(assertion, none_temporal_metadata(), "reason-test", 3.0, reasoning_duration_ms=3.0)

    def normalize_both(self, assertion, episode):
        del episode
        self.both_calls += 1
        return TemporalResult(
            assertion,
            none_temporal_metadata(),
            "both-test",
            8.0,
            reasoning_duration_ms=5.0,
            normalization_duration_ms=3.0,
        )


@dataclass(frozen=True)
class FakeCandidate:
    memory_id: str


def test_memory_pipeline_temporal_gate_routes_and_counts(monkeypatch) -> None:
    assertions = [
        assertion("Priya handles Atlas."),
        assertion("Priya currently handles Atlas."),
        assertion("The user met Kavya yesterday."),
    ]
    gate = FakeGate()
    normalizer = FakeTemporalNormalizer()

    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline._resolve_entities",
        lambda assertion, episode_id: {"subject": {"resolution": "MATCHED", "entity_id": "user-1"}, "semantic_arguments": []},
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.retrieve_vector_candidates",
        lambda canonical_text, top_k=20: SimpleNamespace(
            candidates=[],
            embedding_ms=0.1,
            db_search_ms=0.2,
            total_ms=1.0,
            vector_search_mode="hnsw",
            vector_top_k=top_k,
            vector_returned_count=0,
            hnsw_ef_search=100,
            hnsw_iterative_scan="strict_order",
            hnsw_fallback_used=False,
            hnsw_fallback_reason=None,
        ),
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.retrieve_structured_candidates",
        lambda assertion, entity_resolution, top_k=20: SimpleNamespace(candidates=[], total_ms=1.0),
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.retrieve_graph_candidates",
        lambda assertion, entity_resolution: SimpleNamespace(
            candidates=[],
            graph_retrieval_ms=0.5,
            graph_seed_count=1,
            graph_direct_count=0,
            graph_expanded_count=0,
            graph_returned_count=0,
        ),
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.merge_and_rerank_candidates",
        lambda vector, structured, graph_candidates=None, final_top_k=8: [],
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.reconcile_memory",
        lambda wrapped, final_candidates: ReconciliationDecision(
            op=ReconciliationOp.ADD,
            target_memory_ids=[],
            reason="NO_CANDIDATES_BYPASS",
        ),
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.execute_reconciliation",
        lambda *args, **kwargs: LedgerMutationResult(
            op=ReconciliationOp.ADD,
            target_memory_ids=[],
            created_memory_id="memory-1",
            executed=True,
            event_type="MEMORY_ADDED",
        ),
    )

    result = MemoryPipeline(
        compiler=FakeCompiler(assertions),
        temporal_normalizer=normalizer,
        temporal_gate=gate,
    ).process(episode())

    assert gate.calls == 1
    assert [assertion.canonical_text for assertion in gate.last_assertions] == [
        "Priya handles Atlas.",
        "Priya currently handles Atlas.",
        "The user met Kavya yesterday.",
    ]
    assert normalizer.none_calls == 1
    assert normalizer.reason_calls == 1
    assert normalizer.both_calls == 1
    assert [item["temporal_route"] for item in result.output["assertions"]] == ["NONE", "REASON", "BOTH"]

    temporal = result.output["pipeline_metadata"]["temporal"]
    assert temporal["gate_model_call_count"] == 1
    assert temporal["temporal_route_none_count"] == 1
    assert temporal["temporal_route_reason_count"] == 1
    assert temporal["temporal_route_both_count"] == 1
    assert temporal["temporal_reasoning_call_count"] == 2
    assert temporal["temporal_normalization_call_count"] == 1
    for item in result.output["assertions"]:
        assert item["graph_candidates"] == []
        assert item["graph_retrieval"]["graph_seed_count"] == 1
        assert item["latency"]["graph_retrieval_ms"] == 0.5


def test_memory_pipeline_preserves_calendar_event_for_mutation(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline._resolve_entities",
        lambda assertion, episode_id: {"subject": {"resolution": "MATCHED", "entity_id": "user-1"}, "semantic_arguments": []},
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.retrieve_vector_candidates",
        lambda canonical_text, top_k=20: SimpleNamespace(
            candidates=[],
            embedding_ms=0.1,
            db_search_ms=0.2,
            total_ms=1.0,
            vector_search_mode="hnsw",
            vector_top_k=top_k,
            vector_returned_count=0,
            hnsw_ef_search=100,
            hnsw_iterative_scan="strict_order",
            hnsw_fallback_used=False,
            hnsw_fallback_reason=None,
        ),
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.retrieve_structured_candidates",
        lambda assertion, entity_resolution, top_k=20: SimpleNamespace(candidates=[], total_ms=1.0),
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.retrieve_graph_candidates",
        lambda assertion, entity_resolution: SimpleNamespace(
            candidates=[],
            graph_retrieval_ms=0.5,
            graph_seed_count=1,
            graph_direct_count=0,
            graph_expanded_count=0,
            graph_returned_count=0,
        ),
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.merge_and_rerank_candidates",
        lambda vector, structured, graph_candidates=None, final_top_k=8: [],
    )
    monkeypatch.setattr(
        "kivi_memory.pipeline.memory_pipeline.reconcile_memory",
        lambda wrapped, final_candidates: ReconciliationDecision(
            op=ReconciliationOp.ADD,
            target_memory_ids=[],
            reason="NO_CANDIDATES_BYPASS",
        ),
    )

    def fake_execute(decision, wrapped_assertion, *args, **kwargs):
        del decision, args, kwargs
        captured["wrapped_assertion"] = wrapped_assertion
        return LedgerMutationResult(
            op=ReconciliationOp.ADD,
            target_memory_ids=[],
            created_memory_id="memory-1",
            executed=True,
            event_type="MEMORY_ADDED",
            calendar_event_id="calendar-1",
        )

    monkeypatch.setattr("kivi_memory.pipeline.memory_pipeline.execute_reconciliation", fake_execute)

    result = MemoryPipeline(
        compiler=FakeCompiler([calendar_assertion()]),
        temporal_normalizer=FakeTemporalNormalizer(),
        temporal_gate=OneRouteGate(TemporalRoute.NONE),
    ).process(calendar_episode())

    calendar_event = captured["wrapped_assertion"]["semantic_assertion"]["calendar_event"]
    assert calendar_event["title"] == "Atlas design review"
    assert calendar_event["start_time_text"] == "tomorrow at 3 PM"
    assert result.output["assertions"][0]["semantic_assertion"]["memory_type"] == "CALENDAR_EVENT"
    assert result.output["assertions"][0]["mutation_result"]["calendar_event_id"] == "calendar-1"


def assertion(text: str) -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": text,
            "memory_type": "OTHER",
            "subject": {"text": "user", "entity_type": "person"},
            "semantic_arguments": [],
            "predicate_type": "TEST",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "user",
            "source_spans": [{"message_id": "m1", "text": text}],
        }
    )


class OneRouteGate:
    def __init__(self, route: TemporalRoute) -> None:
        self._route = route

    def route(self, assertions):
        return TemporalGateResult(
            routes={f"A{index + 1}": self._route for index, _ in enumerate(assertions)},
            temporal_gate_ms=1.0,
            gate_prompt_eval_count=0,
            gate_eval_count=0,
            gate_model_call_count=1,
        )


def calendar_assertion() -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": "The user's Atlas design review is tomorrow at 3 PM.",
            "memory_type": "CALENDAR_EVENT",
            "subject": {"text": "user", "entity_type": "person"},
            "semantic_arguments": [
                {"role": "project", "text": "Atlas", "is_entity": True, "entity_type": "PROJECT"},
                {"role": "time", "text": "tomorrow at 3 PM", "is_entity": False, "entity_type": None},
            ],
            "predicate_type": "HAS_SCHEDULED_EVENT",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "user",
            "source_spans": [{"message_id": "m1", "text": "I have the Atlas design review tomorrow at 3 PM."}],
            "calendar_event": {
                "title": "Atlas design review",
                "event_kind": "MEETING",
                "location_text": None,
                "start_time_text": "tomorrow at 3 PM",
                "end_time_text": None,
                "duration_text": None,
                "recurrence_text": None,
                "timezone_text": None,
                "all_day_hint": False,
            },
        }
    )


def episode() -> MemoryEpisode:
    return MemoryEpisode(
        episode_id="ep-1",
        messages=[
            EpisodeMessage(
                message_id="m1",
                role="USER",
                timestamp="2026-09-10T09:00:00+05:30",
                text="Priya handles Atlas. Priya currently handles Atlas. The user met Kavya yesterday.",
            )
        ],
    )


def calendar_episode() -> MemoryEpisode:
    return MemoryEpisode(
        episode_id="calendar-ep-1",
        messages=[
            EpisodeMessage(
                message_id="m1",
                role="USER",
                timestamp="2026-09-06T09:00:00+05:30",
                text="I have the Atlas design review tomorrow at 3 PM.",
            )
        ],
    )
