from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from kivi_memory.common.io import load_compiler_output, load_memory_episode
from kivi_memory.enrichment.temporal import TemporalNormalizer, TemporalNormalizerError
from kivi_memory.enrichment.validator import validate_assertions
from kivi_memory.semantic_compiler.ollama_client import OllamaClientError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run temporal normalization over compiler output.")
    parser.add_argument("--input", default="input.json", type=Path)
    parser.add_argument("--output", default="output.json", type=Path)
    args = parser.parse_args()

    try:
        episode = load_memory_episode(args.input)
        output = load_compiler_output(args.output)
        results = validate_assertions(output.assertions, episode)
        normalizer = TemporalNormalizer()
        for index, result in enumerate(results, start=1):
            print(f"Assertion {index}")
            if not result.validation_report.valid:
                print("Temporal: skipped invalid assertion")
                continue
            temporal = normalizer.normalize(result.assertion, episode).temporal_metadata
            print("Temporal:")
            print(f"  temporal_kind: {temporal.temporal_kind}")
            print(f"  valid_from_hint: {temporal.valid_from_hint}")
            print(f"  valid_to_hint: {temporal.valid_to_hint}")
            print(f"  event_time: {temporal.event_time}")
            print(f"  precision: {temporal.temporal_precision}")
            print(f"  recurrence: {temporal.recurrence}")
            print(f"  recurrence_specifics: {temporal.recurrence_specifics}")
    except (OSError, ValidationError, OllamaClientError, TemporalNormalizerError, ValueError) as exc:
        print(f"Temporal test failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
