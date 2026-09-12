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

from kivi_memory.ledger import MemoryNotFoundError, get_memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch one Canonical Memory Ledger record.")
    parser.add_argument("--memory-id", required=True)
    args = parser.parse_args()

    try:
        memory = get_memory(args.memory_id)
    except MemoryNotFoundError as exc:
        print(f"Memory not found: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Get memory failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(_to_jsonable(memory), indent=2))
    return 0


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
