"""Return models for the add-only memory ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AddMemoryResult:
    memory_id: str
    status: str
    version: int


@dataclass(frozen=True)
class StoredMemory:
    memory: dict[str, Any]
    arguments: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    events: list[dict[str, Any]]


class MemoryNotFoundError(ValueError):
    """Raised when a requested ledger memory does not exist."""
