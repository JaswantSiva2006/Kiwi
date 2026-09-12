"""File helpers for JSON contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kivi_memory.common.schemas import CompilerOutput, MemoryEpisode


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_memory_episode(path: Path) -> MemoryEpisode:
    return MemoryEpisode.model_validate(load_json(path))


def load_compiler_output(path: Path) -> CompilerOutput:
    return CompilerOutput.model_validate(load_json(path))


def write_compiler_output(output: CompilerOutput, path: Path) -> None:
    path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_compiler_output_schema(path: Path) -> None:
    schema = CompilerOutput.model_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
