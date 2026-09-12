from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel

from kivi_memory.common.io import load_memory_episode
from kivi_memory.ledger import add_memory, get_memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Add one pipeline assertion to the Canonical Memory Ledger.")
    parser.add_argument("--pipeline-output", default="pipeline_output.json", type=Path)
    parser.add_argument("--assertion-index", type=int, required=True)
    args = parser.parse_args()

    try:
        pipeline_output = json.loads(args.pipeline_output.read_text(encoding="utf-8"))
        assertion_item = _find_assertion(pipeline_output["assertions"], args.assertion_index)
        input_path = Path(pipeline_output.get("pipeline_metadata", {}).get("input", "input.json"))
        episode = load_memory_episode(input_path)
        result = add_memory(
            assertion_item["semantic_assertion"],
            assertion_item.get("temporal_metadata"),
            assertion_item.get("entity_resolution") or {},
            episode,
            sensitivity_metadata=assertion_item.get("sensitivity_metadata"),
            validation_report=assertion_item.get("validation_report"),
        )
        stored = get_memory(result.memory_id)
    except Exception as exc:
        print(f"Add memory failed: {exc}", file=sys.stderr)
        return 1

    print(f"created memory_id: {result.memory_id}")
    print(json.dumps(_to_jsonable(stored), indent=2))
    return 0


def _find_assertion(assertions: list[dict[str, Any]], index: int) -> dict[str, Any]:
    for assertion in assertions:
        if assertion.get("index") == index:
            return assertion
    raise ValueError(f"assertion index not found: {index}")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
