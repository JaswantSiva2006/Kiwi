from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.corpus_import import convert_jsonl_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ASR/transcript JSONL records into Kivi MemoryEpisode JSONL.")
    parser.add_argument("--input", required=True, help="Input corpus JSONL path")
    parser.add_argument("--output", required=True, help="Output MemoryEpisode JSONL path")
    args = parser.parse_args()

    result = convert_jsonl_file(args.input, args.output)
    if result.errors:
        print(json.dumps({"status": "ERROR", "errors": result.errors}, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "OK",
                "input": args.input,
                "output": args.output,
                "episode_count": len(result.episodes),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
