from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.entity_resolution import resolve_entity_mention


def main() -> int:
    parser = argparse.ArgumentParser(description="Print Kivi entity-resolution decision details.")
    parser.add_argument("--mention", required=True)
    parser.add_argument("--type", dest="entity_type")
    args = parser.parse_args()

    try:
        result = resolve_entity_mention(args.mention, args.entity_type)
    except Exception as exc:
        print(f"Entity resolution failed: {exc}", file=sys.stderr)
        return 1

    print(f"resolution: {result.resolution}")
    print(f"entity_id: {result.entity_id}")
    print(f"reason: {result.reason}")
    print(f"normalized_mention: {result.normalized_mention}")

    if not result.candidates:
        print("candidates: none")
        return 0

    print("candidates:")
    for index, candidate in enumerate(result.candidates, start=1):
        print(f"{index}. {candidate.canonical_name} ({candidate.entity_type})")
        print(f"   entity_id: {candidate.entity_id}")
        print(f"   matched_text: {candidate.matched_text}")
        print(f"   match_type: {candidate.match_type}")
        print(f"   similarity_score: {candidate.similarity_score:.3f}")
        print(f"   type_match: {candidate.type_match}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
