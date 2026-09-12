"""Deterministic projection from canonical ledger rows to graph links."""

from __future__ import annotations

from kivi_memory.graph.models import GraphBackfillResult, GraphProjectionResult, MemoryEntityLink
from kivi_memory.graph.repository import GraphRepository


def project_graph_for_memory(
    memory_id: str,
    repository: GraphRepository | None = None,
) -> GraphProjectionResult:
    """Project subject and resolved entity arguments for one committed memory."""

    repo = repository or GraphRepository()
    source = repo.get_memory_graph_source(memory_id)
    links: list[MemoryEntityLink] = []

    if source.subject_entity_id:
        links.append(
            MemoryEntityLink(
                memory_id=source.memory_id,
                entity_id=source.subject_entity_id,
                link_type="SUBJECT",
                argument_role=None,
                predicate_type=source.predicate_type,
                position=None,
            )
        )

    for argument in source.arguments:
        if not argument.is_entity or argument.entity_id is None:
            continue
        links.append(
            MemoryEntityLink(
                memory_id=source.memory_id,
                entity_id=argument.entity_id,
                link_type="ARGUMENT",
                argument_role=argument.role,
                predicate_type=source.predicate_type,
                position=argument.position,
            )
        )

    inserted = repo.insert_memory_entity_links(links)
    return GraphProjectionResult(
        memory_id=source.memory_id,
        links_considered=len(links),
        links_created=len(inserted),
        subject_links_created=sum(1 for link in inserted if link.link_type == "SUBJECT"),
        argument_links_created=sum(1 for link in inserted if link.link_type == "ARGUMENT"),
    )


def backfill_active_memory_graph(
    repository: GraphRepository | None = None,
    limit: int | None = None,
) -> GraphBackfillResult:
    """Idempotently project graph links for existing ACTIVE memories."""

    repo = repository or GraphRepository()
    memory_ids = repo.list_active_memory_ids(limit=limit)
    links_created = 0
    failures: list[dict[str, str]] = []

    for memory_id in memory_ids:
        try:
            result = project_graph_for_memory(memory_id, repository=repo)
            links_created += result.links_created
        except Exception as exc:
            failures.append({"memory_id": memory_id, "error": str(exc)})

    return GraphBackfillResult(
        total_active_memories=len(memory_ids),
        projected_memories=len(memory_ids) - len(failures),
        links_created=links_created,
        failed=len(failures),
        failures=failures,
    )
