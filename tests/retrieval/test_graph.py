from __future__ import annotations

import pytest

from kivi_memory.retrieval.graph import (
    GRAPH_DISTANCE_DIRECT,
    GRAPH_DISTANCE_ONE_EXPANSION,
    GraphRetrievalResult,
    retrieve_graph_candidates,
)
from kivi_memory.retrieval.models import GraphMemoryCandidate, GraphSignals


class InMemoryGraphRepository:
    def __init__(self, memories: dict[str, dict], links: list[dict]) -> None:
        self.memories = memories
        self.links = links
        self.calls = []

    def retrieve_candidates(self, *, seed_entity_ids, max_bridge_degree, top_k):
        self.calls.append(
            {
                "seed_entity_ids": seed_entity_ids,
                "max_bridge_degree": max_bridge_degree,
                "top_k": top_k,
            }
        )
        direct = self._direct(seed_entity_ids)
        bridge_entities = self._bridge_entities(direct, seed_entity_ids, max_bridge_degree)
        expanded = self._expanded(direct, bridge_entities)
        candidates = list(direct.values()) + list(expanded.values())
        candidates.sort(
            key=lambda candidate: (
                candidate.graph_signals.graph_score,
                candidate.graph_signals.direct_seed_count,
                candidate.graph_signals.bridge_path_count,
                candidate.memory_id,
            ),
            reverse=True,
        )
        return candidates[:top_k]

    def _active_memory(self, memory_id: str) -> dict | None:
        memory = self.memories.get(memory_id)
        if memory and memory["status"] == "ACTIVE":
            return memory
        return None

    def _direct(self, seed_entity_ids: list[str]) -> dict[str, GraphMemoryCandidate]:
        direct = {}
        for link in self.links:
            if link["entity_id"] not in seed_entity_ids:
                continue
            memory = self._active_memory(link["memory_id"])
            if memory is None:
                continue
            existing = direct.get(link["memory_id"])
            seed_count = (existing.graph_signals.direct_seed_count if existing else 0) + 1
            direct[link["memory_id"]] = candidate(
                memory,
                GRAPH_DISTANCE_DIRECT,
                1.0,
                direct_seed_count=seed_count,
            )
        return direct

    def _bridge_entities(
        self,
        direct: dict[str, GraphMemoryCandidate],
        seed_entity_ids: list[str],
        max_bridge_degree: int,
    ) -> list[tuple[str, str]]:
        bridges = []
        for link in self.links:
            if link["memory_id"] not in direct:
                continue
            entity_id = link["entity_id"]
            if entity_id in seed_entity_ids or entity_id == "user-1":
                continue
            if self._active_degree(entity_id) > max_bridge_degree:
                continue
            bridges.append((entity_id, link["memory_id"]))
        return sorted(set(bridges))

    def _active_degree(self, entity_id: str) -> int:
        return len(
            {
                link["memory_id"]
                for link in self.links
                if link["entity_id"] == entity_id and self._active_memory(link["memory_id"])
            }
        )

    def _expanded(
        self,
        direct: dict[str, GraphMemoryCandidate],
        bridge_entities: list[tuple[str, str]],
    ) -> dict[str, GraphMemoryCandidate]:
        expanded = {}
        path_counts = {}
        for bridge_entity_id, via_memory_id in bridge_entities:
            for link in self.links:
                if link["entity_id"] != bridge_entity_id:
                    continue
                memory_id = link["memory_id"]
                if memory_id in direct:
                    continue
                memory = self._active_memory(memory_id)
                if memory is None:
                    continue
                path_counts.setdefault(memory_id, set()).add((bridge_entity_id, via_memory_id))
                expanded[memory_id] = candidate(
                    memory,
                    GRAPH_DISTANCE_ONE_EXPANSION,
                    0.5,
                    bridge_path_count=len(path_counts[memory_id]),
                )
        return expanded


def test_seed_entity_retrieves_direct_memory() -> None:
    repo = graph_repo(
        memories=[memory("m1", "Priya handles Atlas.")],
        links=[link("m1", "priya-1"), link("m1", "atlas-1")],
    )

    result = retrieve_graph_candidates(assertion(), entity_resolution(subject_id="priya-1"), repository=repo)

    assert [candidate.memory_id for candidate in result.candidates] == ["m1"]
    assert result.candidates[0].graph_signals.graph_distance == GRAPH_DISTANCE_DIRECT
    assert result.graph_direct_count == 1


def test_one_expansion_retrieves_memory_through_related_entity() -> None:
    repo = graph_repo(
        memories=[
            memory("m1", "Priya handles Atlas."),
            memory("m2", "Atlas uses Kafka."),
            memory("m3", "Kafka migration happens in November."),
        ],
        links=[
            link("m1", "priya-1"),
            link("m1", "atlas-1"),
            link("m2", "atlas-1"),
            link("m2", "kafka-1"),
            link("m3", "kafka-1"),
        ],
    )

    result = retrieve_graph_candidates(assertion(), entity_resolution(subject_id="priya-1"), repository=repo)

    by_id = {candidate.memory_id: candidate for candidate in result.candidates}
    assert by_id["m1"].graph_signals.graph_distance == GRAPH_DISTANCE_DIRECT
    assert by_id["m2"].graph_signals.graph_distance == GRAPH_DISTANCE_ONE_EXPANSION
    assert "m3" not in by_id


def test_superseded_and_retracted_memories_are_not_returned() -> None:
    repo = graph_repo(
        memories=[
            memory("m1", "old", status="SUPERSEDED"),
            memory("m2", "correction", status="RETRACTED"),
        ],
        links=[link("m1", "priya-1"), link("m2", "priya-1")],
    )

    result = retrieve_graph_candidates(assertion(), entity_resolution(subject_id="priya-1"), repository=repo)

    assert result.candidates == []


def test_unresolved_entities_create_no_seed() -> None:
    repo = graph_repo(
        memories=[memory("m1", "Priya handles Atlas.")],
        links=[link("m1", "priya-1")],
    )

    result = retrieve_graph_candidates(assertion(), {"subject": {"resolution": "AMBIGUOUS", "entity_id": None}}, repository=repo)

    assert result.candidates == []
    assert repo.calls == []


def test_multiple_paths_to_same_memory_produce_one_candidate() -> None:
    repo = graph_repo(
        memories=[
            memory("m1", "Priya handles Atlas with Rohit."),
            memory("m2", "Atlas and Rohit are linked."),
        ],
        links=[
            link("m1", "priya-1"),
            link("m1", "atlas-1"),
            link("m1", "rohit-1"),
            link("m2", "atlas-1"),
            link("m2", "rohit-1"),
        ],
    )

    result = retrieve_graph_candidates(assertion(), entity_resolution(subject_id="priya-1"), repository=repo)
    expanded = {candidate.memory_id: candidate for candidate in result.candidates}["m2"]

    assert [candidate.memory_id for candidate in result.candidates].count("m2") == 1
    assert expanded.graph_signals.bridge_path_count == 2


def test_direct_and_expanded_path_to_same_memory_resolves_to_direct() -> None:
    repo = graph_repo(
        memories=[
            memory("m1", "Priya handles Atlas."),
            memory("m2", "Priya reviews Atlas."),
        ],
        links=[
            link("m1", "priya-1"),
            link("m1", "atlas-1"),
            link("m2", "priya-1"),
            link("m2", "atlas-1"),
        ],
    )

    result = retrieve_graph_candidates(assertion(), entity_resolution(subject_id="priya-1"), repository=repo)

    assert all(candidate.graph_signals.graph_distance == GRAPH_DISTANCE_DIRECT for candidate in result.candidates)


def test_high_degree_bridge_entities_are_not_expanded() -> None:
    memories = [memory("m1", "Priya handles Atlas."), memory("m2", "Atlas uses Kafka.")]
    memories.extend(memory(f"hub-{index}", f"Atlas hub memory {index}") for index in range(3))
    links = [link("m1", "priya-1"), link("m1", "atlas-1"), link("m2", "atlas-1")]
    links.extend(link(f"hub-{index}", "atlas-1") for index in range(3))
    repo = graph_repo(memories=memories, links=links)

    result = retrieve_graph_candidates(
        assertion(),
        entity_resolution(subject_id="priya-1"),
        max_bridge_degree=2,
        repository=repo,
    )

    assert [candidate.memory_id for candidate in result.candidates] == ["m1"]


def test_current_user_hub_does_not_explode_traversal() -> None:
    repo = graph_repo(
        memories=[
            memory("m1", "The user works with Priya."),
            memory("m2", "Unrelated user memory."),
            memory("m3", "Priya handles Atlas."),
        ],
        links=[
            link("m1", "user-1"),
            link("m1", "priya-1"),
            link("m2", "user-1"),
            link("m3", "priya-1"),
        ],
    )

    result = retrieve_graph_candidates(
        assertion(),
        {
            "subject": {
                "resolution": "MATCHED",
                "entity_id": "user-1",
                "reason": "CURRENT_USER_SELF_REFERENCE",
                "normalized_mention": "user",
            },
            "semantic_arguments": [
                {"result": {"resolution": "MATCHED", "entity_id": "priya-1", "reason": "UNIQUE_EXACT_TYPE_MATCH"}},
            ],
        },
        repository=repo,
    )

    assert repo.calls[0]["seed_entity_ids"] == ["priya-1"]
    assert {candidate.memory_id for candidate in result.candidates} == {"m1", "m3"}


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_fails(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve_graph_candidates(assertion(), entity_resolution(subject_id="priya-1"), top_k=top_k)


def assertion() -> dict:
    return {
        "canonical_text": "Priya handles Atlas.",
        "memory_type": "PROJECT_GOAL_TOPIC",
        "subject": {"text": "Priya", "entity_type": "person"},
        "semantic_arguments": [
            {"role": "project", "text": "Atlas", "is_entity": True, "entity_type": "project"},
        ],
        "predicate_type": "RESPONSIBLE_FOR",
        "modality": "FACT",
        "polarity": "POSITIVE",
        "certainty": "CERTAIN",
        "explicitness": "EXPLICIT",
        "attributed_to": "user",
        "source_spans": [{"message_id": "m1", "text": "Priya handles Atlas."}],
    }


def entity_resolution(subject_id: str | None = None, argument_id: str | None = None) -> dict:
    return {
        "subject": {"resolution": "MATCHED", "entity_id": subject_id},
        "semantic_arguments": [
            {"result": {"resolution": "MATCHED", "entity_id": argument_id}} if argument_id else {}
        ],
    }


def graph_repo(memories: list[dict], links: list[dict]) -> InMemoryGraphRepository:
    return InMemoryGraphRepository({row["memory_id"]: row for row in memories}, links)


def memory(memory_id: str, canonical_text: str, status: str = "ACTIVE") -> dict:
    return {
        "memory_id": memory_id,
        "canonical_text": canonical_text,
        "subject_entity_id": None,
        "predicate_type": "RESPONSIBLE_FOR",
        "memory_type": "PROJECT_GOAL_TOPIC",
        "status": status,
    }


def link(memory_id: str, entity_id: str) -> dict:
    return {"memory_id": memory_id, "entity_id": entity_id}


def candidate(
    memory: dict,
    graph_distance: str,
    graph_score: float,
    direct_seed_count: int = 0,
    bridge_path_count: int = 0,
) -> GraphMemoryCandidate:
    return GraphMemoryCandidate(
        memory_id=memory["memory_id"],
        canonical_text=memory["canonical_text"],
        graph_signals=GraphSignals(
            graph_distance=graph_distance,
            graph_score=graph_score,
            direct_seed_count=direct_seed_count,
            bridge_path_count=bridge_path_count,
        ),
        subject_entity_id=memory["subject_entity_id"],
        predicate_type=memory["predicate_type"],
        memory_type=memory["memory_type"],
        status=memory["status"],
    )
