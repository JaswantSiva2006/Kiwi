from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from kivi_memory.common.io import load_memory_episode
from kivi_memory.pipeline.write_pipeline import WritePipeline
from kivi_memory.semantic_compiler.compiler import SemanticCompilerError
from kivi_memory.semantic_compiler.ollama_client import OllamaClientError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the implemented Kivi write pipeline.")
    parser.add_argument("--input", default="input.json", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        episode = load_memory_episode(args.input)
        result = WritePipeline().process(episode)
    except (OSError, ValidationError, OllamaClientError, SemanticCompilerError, ValueError) as exc:
        print(f"Write pipeline failed: {exc}", file=sys.stderr)
        return 1

    invalid = sum(1 for item in result.enriched_assertions if not item.validation_report.valid)
    warnings = sum(
        1
        for item in result.enriched_assertions
        for issue in item.validation_report.issues
        if issue.severity == "WARNING"
    )
    temporal_count = sum(1 for item in result.enriched_assertions if item.temporal_metadata is not None)
    print(f"Episode: {result.episode_id}")
    print(f"Compiler: PASS ({len(result.compiler_result.output.assertions)} assertions)")
    print(f"Deterministic Validator: {len(result.enriched_assertions) - invalid} valid, {invalid} invalid")
    print(f"Temporal Normalizer: {temporal_count} processed")
    print(f"Warnings: {warnings}")
    print("Storage: not implemented")
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
