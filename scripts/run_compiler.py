from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from kivi_memory.semantic_compiler.compiler import SemanticCompiler, SemanticCompilerError
from kivi_memory.common.config import load_config_from_env
from kivi_memory.ingestion.episode_formatter import format_memory_episode
from kivi_memory.common.io import load_memory_episode, write_compiler_output
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local Kivi semantic compiler.")
    parser.add_argument("--input", default="input.json", type=Path)
    parser.add_argument("--output", default="output.json", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--ollama-url")
    parser.add_argument("--show-prompt-input", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config_from_env()
    if args.model:
        config = config.__class__(**{**config.__dict__, "model": args.model})
    if args.ollama_url:
        config = config.__class__(**{**config.__dict__, "ollama_base_url": args.ollama_url})

    try:
        episode = load_memory_episode(args.input)
        if args.show_prompt_input:
            print(format_memory_episode(episode))
            print()

        client = OllamaClient(config.ollama_base_url, config.timeout_seconds)
        client.health()
        compiler = SemanticCompiler(client=client, config=config)
        result = compiler.compile(episode)
        write_compiler_output(result.output, args.output)
    except (OSError, ValidationError, OllamaClientError, SemanticCompilerError, ValueError) as exc:
        print(f"Compiler failed: {exc}", file=sys.stderr)
        return 1

    print(f"Model: {result.model_name}")
    print(f"Episode: {episode.episode_id}")
    print(f"Assertions: {len(result.output.assertions)}")
    print(f"Validation: {'PASS' if result.validation_report.valid else 'FAIL'}")
    print(f"Inference: {result.inference_duration_ms} ms")
    print(f"Attempts: {result.attempt_count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
