"""Convert transcript/ASR corpus records into existing MemoryEpisode JSONL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kivi_memory.common.config import DEFAULT_LOCALE, DEFAULT_TIMEZONE
from kivi_memory.common.schemas import MemoryEpisode


class CorpusConversionError(ValueError):
    """Raised when a corpus record cannot be converted."""


class CorpusRecord(BaseModel):
    """Input JSONL record accepted by the corpus importer."""

    model_config = ConfigDict(extra="allow")

    record_id: str = Field(min_length=1)
    raw_asr: str = Field(min_length=1)
    formatted_text: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("record_id", "raw_asr", "formatted_text", "timestamp")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = str(value)
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("metadata")
    @classmethod
    def metadata_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("metadata must be an object")
        return value


@dataclass(frozen=True)
class CorpusConversionResult:
    episodes: list[MemoryEpisode]
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CorpusMemoryEpisodeBuilder:
    """Build one compiler-safe episode per corpus record."""

    session_id: str = "corpus-import"

    def build_all(self, records: list[CorpusRecord]) -> list[MemoryEpisode]:
        return [self.build(records, index) for index in range(len(records))]

    def build(self, records: list[CorpusRecord], target_index: int) -> MemoryEpisode:
        if target_index < 0 or target_index >= len(records):
            raise IndexError("target_index is outside records")
        target = records[target_index]
        message = {
            "message_id": _message_id(target.record_id, role="user"),
            "role": "USER",
            "timestamp": target.timestamp,
            "text": target.formatted_text,
            "memory_eligible": True,
            "context_only": False,
            "source_thread_episode_id": target.record_id,
            "source_stream_id": str(target_index),
        }

        return MemoryEpisode.model_validate(
            {
                "episode_id": _episode_id(target.record_id),
                "session_id": self.session_id,
                "timezone": _metadata_text(target.metadata, "timezone", DEFAULT_TIMEZONE),
                "locale": _metadata_text(target.metadata, "locale", DEFAULT_LOCALE),
                "messages": [message],
                "tool_context": [
                    {
                        "tool_name": "corpus_import",
                        "source_message_id": _message_id(target.record_id, role="user"),
                        "content": _provenance_payload(target_index, target),
                    }
                ],
                "grounding_context": None,
            }
        )


def read_corpus_jsonl(path: str | Path) -> tuple[list[CorpusRecord], list[dict[str, Any]]]:
    records: list[CorpusRecord] = []
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            records.append(CorpusRecord.model_validate(raw))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            errors.append({"line": line_number, "error": _error_text(exc)})
    return records, errors


def convert_jsonl_file(input_path: str | Path, output_path: str | Path) -> CorpusConversionResult:
    records, errors = read_corpus_jsonl(input_path)
    if errors:
        return CorpusConversionResult(episodes=[], errors=errors)
    builder = CorpusMemoryEpisodeBuilder()
    episodes: list[MemoryEpisode] = []
    for index, record in enumerate(records):
        try:
            episodes.append(builder.build(records, index))
        except Exception as exc:
            errors.append({"record_id": record.record_id, "error": _error_text(exc)})
    if errors:
        return CorpusConversionResult(episodes=[], errors=errors)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(episode.model_dump_json(exclude_none=True) + "\n")
    return CorpusConversionResult(episodes=episodes, errors=[])


def _episode_id(record_id: str) -> str:
    return f"corpus-{_stable_slug(record_id)}"


def _message_id(record_id: str, *, role: str) -> str:
    return f"corpus-{_stable_slug(record_id)}-{role}"


def _stable_slug(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    safe = "-".join(part for part in safe.split("-") if part)
    if safe:
        return safe[:80]
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _metadata_text(metadata: dict[str, Any], key: str, default: str) -> str:
    value = metadata.get(key)
    if value is None or not str(value).strip():
        return default
    return str(value)


def _provenance_payload(target_index: int, record: CorpusRecord) -> str:
    payload = {
        "target_record_id": record.record_id,
        "position": target_index,
        "raw_asr": record.raw_asr,
        "timestamp": record.timestamp,
        "metadata": record.metadata,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error_text(exc: Exception) -> str:
    return " ".join(str(exc).split())
