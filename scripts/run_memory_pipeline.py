from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.common.io import load_memory_episode
from kivi_memory.pipeline import MemoryPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete Kivi memory pipeline built so far.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("memory_pipeline_output.json"), type=Path)
    args = parser.parse_args()

    episode = load_memory_episode(args.input)
    result = MemoryPipeline().process(episode)
    args.output.write_text(json.dumps(result.output, indent=2) + "\n", encoding="utf-8")
    print(f"Memory pipeline output written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
