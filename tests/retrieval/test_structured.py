from __future__ import annotations

import pytest

from kivi_memory.retrieval import retrieve_structured_candidates
from kivi_memory.retrieval.structured import MAX_STRUCTURED_RETRIEVAL_TOP_K


class CapturingStructuredRepository:
    def __init__(self) -> None:
        self.kwargs = None

    def retrieve_candidates(self, **kwargs):
        self.kwargs = kwargs
        return []


def test_structured_retrieval_extracts_resolved_subject_and_argument_entities() -> None:
    repo = CapturingStructuredRepository()

    retrieve_structured_candidates(assertion(), entity_resolution(), repository=repo)

    assert repo.kwargs["subject_entity_id"] == "subject-1"
    assert repo.kwargs["argument_entity_ids"] == ["asset-1"]
    assert repo.kwargs["predicate_type"] == "OWNS"
    assert repo.kwargs["memory_type"] == "PROJECT_GOAL_TOPIC"


def test_structured_retrieval_ignores_ambiguous_and_non_entity_arguments() -> None:
    repo = CapturingStructuredRepository()

    retrieve_structured_candidates(assertion(), ambiguous_entity_resolution(), repository=repo)

    assert repo.kwargs["argument_entity_ids"] == []


@pytest.mark.parametrize("top_k", [0, -1, MAX_STRUCTURED_RETRIEVAL_TOP_K + 1])
def test_invalid_top_k_fails(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve_structured_candidates(assertion(), entity_resolution(), top_k=top_k)


def assertion() -> dict:
    return {
        "canonical_text": "Riya owns the billing dashboard.",
        "memory_type": "PROJECT_GOAL_TOPIC",
        "subject": {"text": "Riya", "entity_type": "person"},
        "semantic_arguments": [
            {"role": "asset", "text": "the billing dashboard", "is_entity": True, "entity_type": "dashboard"},
            {"role": "time", "text": "today", "is_entity": False, "entity_type": None},
        ],
        "predicate_type": "OWNS",
        "modality": "FACT",
        "polarity": "POSITIVE",
        "certainty": "CERTAIN",
        "explicitness": "EXPLICIT",
        "attributed_to": "user",
        "source_spans": [{"message_id": "m1", "text": "Riya owns the billing dashboard."}],
    }


def entity_resolution() -> dict:
    return {
        "subject": {"resolution": "MATCHED", "entity_id": "subject-1"},
        "semantic_arguments": [
            {"role": "asset", "result": {"resolution": "MATCHED", "entity_id": "asset-1"}},
        ],
    }


def ambiguous_entity_resolution() -> dict:
    return {
        "subject": {"resolution": "MATCHED", "entity_id": "subject-1"},
        "semantic_arguments": [
            {"role": "asset", "result": {"resolution": "AMBIGUOUS", "entity_id": None}},
        ],
    }
