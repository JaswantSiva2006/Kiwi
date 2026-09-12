from __future__ import annotations

import pytest

from kivi_memory.graph import backfill_active_memory_graph, project_graph_for_memory
from kivi_memory.graph.models import GraphSourceArgument, GraphSourceMemory, MemoryEntityLink
from kivi_memory.ledger import MemoryNotFoundError


class FakeGraphRepository:
    def __init__(
        self,
        sources: dict[str, GraphSourceMemory],
        active_memory_ids: list[str] | None = None,
    ) -> None:
        self.sources = sources
        self.active_memory_ids = active_memory_ids or list(sources)
        self.links: list[MemoryEntityLink] = []
        self.keys: set[tuple] = set()

    def get_memory_graph_source(self, memory_id: str) -> GraphSourceMemory:
        try:
            return self.sources[memory_id]
        except KeyError:
            raise MemoryNotFoundError(f"memory does not exist: {memory_id}") from None

    def insert_memory_entity_links(self, links: list[MemoryEntityLink]) -> list[MemoryEntityLink]:
        inserted = []
        for link in links:
            key = (
                link.memory_id,
                link.entity_id,
                link.link_type,
                link.argument_role,
                link.predicate_type,
                link.position,
            )
            if key in self.keys:
                continue
            self.keys.add(key)
            self.links.append(link)
            inserted.append(link)
        return inserted

    def list_active_memory_ids(self, limit: int | None = None) -> list[str]:
        if limit is None:
            return list(self.active_memory_ids)
        return list(self.active_memory_ids[:limit])


def test_subject_entity_creates_subject_link() -> None:
    repo = FakeGraphRepository(
        {"m1": source_memory(subject_entity_id="e1", arguments=[])}
    )

    result = project_graph_for_memory("m1", repository=repo)

    assert result.subject_links_created == 1
    assert repo.links == [
        MemoryEntityLink(
            memory_id="m1",
            entity_id="e1",
            link_type="SUBJECT",
            argument_role=None,
            predicate_type="RESPONSIBLE_FOR",
            position=None,
        )
    ]


def test_resolved_entity_argument_creates_argument_link() -> None:
    repo = FakeGraphRepository(
        {
            "m1": source_memory(
                subject_entity_id=None,
                arguments=[argument("project", True, "e2", 0)],
            )
        }
    )

    result = project_graph_for_memory("m1", repository=repo)

    assert result.argument_links_created == 1
    assert repo.links[0].link_type == "ARGUMENT"
    assert repo.links[0].argument_role == "project"
    assert repo.links[0].position == 0


def test_unresolved_and_non_entity_arguments_create_no_graph_links() -> None:
    repo = FakeGraphRepository(
        {
            "m1": source_memory(
                subject_entity_id=None,
                arguments=[
                    argument("project", True, None, 0),
                    argument("date", False, "should-not-link", 1),
                ],
            )
        }
    )

    result = project_graph_for_memory("m1", repository=repo)

    assert result.links_considered == 0
    assert repo.links == []


def test_multiple_entities_in_one_memory_create_multiple_links() -> None:
    repo = FakeGraphRepository(
        {
            "m1": source_memory(
                subject_entity_id="e1",
                arguments=[
                    argument("project", True, "e2", 0),
                    argument("collaborator", True, "e3", 1),
                    argument("schedule", False, None, 2),
                ],
            )
        }
    )

    result = project_graph_for_memory("m1", repository=repo)

    assert result.links_created == 3
    assert [(link.entity_id, link.link_type, link.argument_role) for link in repo.links] == [
        ("e1", "SUBJECT", None),
        ("e2", "ARGUMENT", "project"),
        ("e3", "ARGUMENT", "collaborator"),
    ]


def test_missing_memory_fails_cleanly() -> None:
    with pytest.raises(MemoryNotFoundError, match="memory does not exist"):
        project_graph_for_memory("missing", repository=FakeGraphRepository({}))


def test_repeated_projection_is_idempotent() -> None:
    repo = FakeGraphRepository(
        {
            "m1": source_memory(
                subject_entity_id="e1",
                arguments=[argument("project", True, "e2", 0)],
            )
        }
    )

    first = project_graph_for_memory("m1", repository=repo)
    second = project_graph_for_memory("m1", repository=repo)

    assert first.links_created == 2
    assert second.links_created == 0
    assert len(repo.links) == 2


def test_backfill_active_memories_is_idempotent() -> None:
    repo = FakeGraphRepository(
        {
            "m1": source_memory(subject_entity_id="e1", arguments=[]),
            "m2": source_memory(memory_id="m2", subject_entity_id=None, arguments=[argument("owner", True, "e2", 0)]),
        },
        active_memory_ids=["m1", "m2"],
    )

    first = backfill_active_memory_graph(repository=repo)
    second = backfill_active_memory_graph(repository=repo)

    assert first.total_active_memories == 2
    assert first.projected_memories == 2
    assert first.links_created == 2
    assert first.failed == 0
    assert second.links_created == 0
    assert len(repo.links) == 2


def source_memory(
    memory_id: str = "m1",
    subject_entity_id: str | None = "e1",
    arguments: list[GraphSourceArgument] | None = None,
) -> GraphSourceMemory:
    return GraphSourceMemory(
        memory_id=memory_id,
        predicate_type="RESPONSIBLE_FOR",
        subject_entity_id=subject_entity_id,
        arguments=arguments or [],
    )


def argument(role: str, is_entity: bool, entity_id: str | None, position: int) -> GraphSourceArgument:
    return GraphSourceArgument(
        role=role,
        is_entity=is_entity,
        entity_id=entity_id,
        position=position,
    )
