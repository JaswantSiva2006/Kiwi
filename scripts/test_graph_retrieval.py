from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.entity_resolution import resolve_entity_mention
from kivi_memory.retrieval import retrieve_graph_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug graph retrieval from one resolved mention.")
    parser.add_argument("--mention", required=True)
    parser.add_argument("--type", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    resolution = resolve_entity_mention(args.mention, args.type)
    assertion = {
        "canonical_text": args.mention,
        "memory_type": "OTHER",
        "subject": {"text": args.mention, "entity_type": args.type},
        "semantic_arguments": [],
        "predicate_type": "RELATED_TO",
        "modality": "FACT",
        "polarity": "POSITIVE",
        "certainty": "CERTAIN",
        "explicitness": "EXPLICIT",
        "attributed_to": "user",
        "source_spans": [{"message_id": "debug", "text": args.mention}],
    }
    result = retrieve_graph_candidates(
        assertion,
        {"subject": resolution, "semantic_arguments": []},
        top_k=args.top_k,
    )
    print(json.dumps({"entity_resolution": resolution, "graph_retrieval": result}, default=_json_default, indent=2))


def _json_default(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


if __name__ == "__main__":
    main()
