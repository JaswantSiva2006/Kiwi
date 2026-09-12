from __future__ import annotations

import json

from kivi_memory.common.schemas import CandidateSemanticAssertion
from kivi_memory.enrichment.temporal_gate import (
    TEMPORAL_GATE_MODEL,
    TEMPORAL_GATE_SYSTEM_PROMPT,
    TEMPORAL_GATE_THINK,
    TEMPORAL_GATE_TEMPERATURE,
    TemporalRoute,
    TemporalRoutingGate,
    TemporalGateOutput,
    format_temporal_gate_input,
    validate_gate_routes,
)


class FakeOllamaClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls = []

    def _post(self, path: str, payload: dict):
        self.calls.append({"path": path, "payload": payload})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {
            "message": {"content": json.dumps(response)},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }


def test_gate_input_contains_only_ids_and_canonical_text() -> None:
    assertions = [
        assertion("The user met Kavya yesterday."),
        assertion("Riya owns the billing dashboard."),
    ]

    text = format_temporal_gate_input(["A1", "A2"], assertions)

    assert text == "A1|The user met Kavya yesterday.\nA2|Riya owns the billing dashboard."
    assert "PROJECT_GOAL_TOPIC" not in text
    assert "predicate_type" not in text
    assert "message_id" not in text


def test_gate_uses_qwen_4b_think_false_temperature_zero() -> None:
    client = FakeOllamaClient(
        [{"routes": [{"id": "A1", "route": "NONE"}]}]
    )
    gate = TemporalRoutingGate(client=client)

    result = gate.route([assertion("Priya handles Atlas.")])

    payload = client.calls[0]["payload"]
    assert payload["model"] == TEMPORAL_GATE_MODEL == "qwen3.5:4b"
    assert payload["think"] is TEMPORAL_GATE_THINK is False
    assert payload["options"]["temperature"] == TEMPORAL_GATE_TEMPERATURE == 0
    assert payload["messages"][0]["content"] == TEMPORAL_GATE_SYSTEM_PROMPT
    assert result.routes["A1"] == TemporalRoute.NONE
    assert result.gate_model_call_count == 1


def test_validate_gate_routes_rejects_missing_duplicate_and_invented_ids() -> None:
    assert validate_gate_routes(
        TemporalGateOutput.model_validate({"routes": [{"id": "A1", "route": "BOTH"}]}),
        ["A1"],
    ) == {"A1": TemporalRoute.BOTH}

    for bad in [
        {"routes": [{"id": "A1", "route": "BOTH"}, {"id": "A1", "route": "NONE"}]},
        {"routes": [{"id": "A2", "route": "BOTH"}]},
        {"routes": [{"id": "A1", "route": "BOTH"}]},
    ]:
        ids = ["A1", "A2"] if bad["routes"][0]["id"] == "A1" else ["A1"]
        try:
            validate_gate_routes(TemporalGateOutput.model_validate(bad), ids)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid gate output was accepted")


def test_gate_retries_once_then_falls_back_to_both() -> None:
    client = FakeOllamaClient([
        {"routes": [{"id": "A1", "route": "NONE"}]},
        {"routes": [{"id": "A2", "route": "NONE"}]},
    ])
    gate = TemporalRoutingGate(client=client)

    result = gate.route([assertion("A"), assertion("B")])

    assert result.routes == {"A1": TemporalRoute.BOTH, "A2": TemporalRoute.BOTH}
    assert result.fallback_used is True
    assert result.gate_model_call_count == 2


def test_routing_examples_are_accepted_from_batched_gate_output() -> None:
    examples = [
        ("The user met Kavya yesterday.", "BOTH"),
        ("The design review is tomorrow at 3:30 PM.", "BOTH"),
        ("The user worked on Orion from January 2024 until March 2025.", "BOTH"),
        ("The vendor contract expires on December 31, 2026.", "BOTH"),
        ("The migration is expected around November 2026.", "BOTH"),
        ("The annual audit happens every July.", "REASON"),
        ("The user pays rent on the 5th every month.", "REASON"),
        ("The user no longer likes coffee.", "REASON"),
        ("Priya currently handles Atlas.", "REASON"),
        ("Riya owns the billing dashboard.", "NONE"),
        ("Priya handles Atlas.", "NONE"),
        ("The user likes poker.", "NONE"),
        ("The user pays rent every month starting next January.", "BOTH"),
        ("Priya currently handles Atlas until December.", "BOTH"),
    ]
    expected = [{"id": f"A{index + 1}", "route": route} for index, (_, route) in enumerate(examples)]
    gate = TemporalRoutingGate(client=FakeOllamaClient([{"routes": expected}]))

    result = gate.route([assertion(text) for text, _ in examples])

    assert [result.routes[f"A{index + 1}"].value for index in range(len(examples))] == [
        route for _, route in examples
    ]


def assertion(text: str) -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": text,
            "memory_type": "PROJECT_GOAL_TOPIC",
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
