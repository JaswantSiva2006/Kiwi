"""Run derived projections after canonical ledger writes have committed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kivi_memory.embeddings.indexer import index_memory

Projection = Callable[[str], Any]


@dataclass(frozen=True)
class PostCommitProjectionResult:
    graph_projection: Any = None
    graph_projection_error: str | None = None
    embedding_projection: Any = None
    embedding_projection_error: str | None = None


def run_post_commit_projections(
    memory_id: str,
    *,
    graph_projector: Projection | None = None,
    embedding_indexer: Projection | None = None,
) -> PostCommitProjectionResult:
    """Project derived state without affecting the committed canonical memory."""

    graph_projection = None
    graph_projection_error = None
    embedding_projection = None
    embedding_projection_error = None

    try:
        graph_projection = (graph_projector or _default_graph_projector())(memory_id)
    except Exception as exc:
        graph_projection_error = str(exc)

    try:
        embedding_projection = (embedding_indexer or index_memory)(memory_id)
    except Exception as exc:
        embedding_projection_error = str(exc)

    return PostCommitProjectionResult(
        graph_projection=graph_projection,
        graph_projection_error=graph_projection_error,
        embedding_projection=embedding_projection,
        embedding_projection_error=embedding_projection_error,
    )


def _default_graph_projector() -> Projection:
    from kivi_memory.graph.projector import project_graph_for_memory

    return project_graph_for_memory
