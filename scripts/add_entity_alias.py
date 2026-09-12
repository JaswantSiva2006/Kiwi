from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.entity_resolution import EntityNotFoundError, add_entity_alias


def main() -> int:
    parser = argparse.ArgumentParser(description="Add an alias to an existing Kivi entity.")
    parser.add_argument("--entity-id", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--source", dest="alias_source")
    parser.add_argument("--source-episode-id")
    args = parser.parse_args()

    try:
        alias = add_entity_alias(
            args.entity_id,
            args.alias,
            args.confidence,
            args.alias_source,
            args.source_episode_id,
        )
    except EntityNotFoundError as exc:
        print(f"Add alias failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Add alias failed: {exc}", file=sys.stderr)
        return 1

    print(f"Alias record: {alias.alias}")
    print(f"alias_id: {alias.alias_id}")
    print(f"entity_id: {alias.entity_id}")
    print(f"normalized_alias: {alias.normalized_alias}")
    print(f"confidence: {alias.confidence}")
    print(f"alias_source: {alias.alias_source}")
    print(f"source_episode_id: {alias.source_episode_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
