from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel, ValidationError

from kivi_memory.common.config import load_config_from_env
from kivi_memory.common.io import load_memory_episode
from kivi_memory.common.schemas import CandidateSemanticAssertion, CompilerOutput, SemanticArgument
from kivi_memory.enrichment.temporal import TemporalNormalizer, TemporalNormalizerError
from kivi_memory.enrichment.validator import validate_assertions
from kivi_memory.entity_resolution import resolve_entity_mention, resolve_or_create_entity
from kivi_memory.ingestion.episode_formatter import format_memory_episode
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError
from kivi_memory.semantic_compiler.prompt import SYSTEM_PROMPT

SEMANTIC_COMPILER_MODEL = "qwen3.5:9b"
SEMANTIC_COMPILER_THINK = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Kivi pipeline stages implemented so far.")
    parser.add_argument("--input", default="input.json", type=Path)
    parser.add_argument("--output", default="pipeline_output.json", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config_from_env()

    try:
        episode = load_memory_episode(args.input)
        formatted_episode = format_memory_episode(episode)
        compiler_result = _run_semantic_compiler(episode, formatted_episode, config)
    except (OSError, ValidationError, OllamaClientError, ValueError) as exc:
        print(f"Pipeline failed before assertion enrichment: {exc}", file=sys.stderr)
        return 1

    temporal_normalizer = TemporalNormalizer(config=config)
    validated_assertions = validate_assertions(compiler_result.output.assertions, episode)
    assertion_outputs = []

    for index, validated in enumerate(validated_assertions):
        assertion = validated.assertion
        assertion_output: dict[str, Any] = {
            "index": index,
            "semantic_assertion": _to_jsonable(assertion),
            "validation_report": _to_jsonable(validated.validation_report),
            "temporal_metadata": None,
            "sensitivity_metadata": None,
            "entity_resolution": {
                "subject": None,
                "semantic_arguments": [],
            },
            "enrichment_errors": [],
        }

        if not validated.validation_report.valid:
            assertion_outputs.append(assertion_output)
            continue

        try:
            temporal_result = temporal_normalizer.normalize(assertion, episode)
            assertion_output["temporal_metadata"] = _to_jsonable(temporal_result.temporal_metadata)
        except (TemporalNormalizerError, OllamaClientError, ValidationError, ValueError) as exc:
            assertion_output["enrichment_errors"].append(
                {
                    "stage": "temporal_normalizer",
                    "error": str(exc),
                }
            )

        assertion_output["entity_resolution"]["subject"] = _resolve_entity(
            text=assertion.subject.text,
            entity_type=assertion.subject.entity_type,
            source_episode_id=episode.episode_id,
            role="subject",
            errors=assertion_output["enrichment_errors"],
        )

        for argument in assertion.semantic_arguments:
            if not argument.is_entity:
                continue
            assertion_output["entity_resolution"]["semantic_arguments"].append(
                {
                    "role": argument.role,
                    "result": _resolve_argument(argument, assertion, episode.episode_id, assertion_output["enrichment_errors"]),
                }
            )

        assertion_outputs.append(assertion_output)

    output = {
        "episode_id": episode.episode_id,
        "models": {
            "semantic_compiler": compiler_result.model_name,
            "semantic_compiler_think": SEMANTIC_COMPILER_THINK,
            "temporal_reasoning": config.temporal_reasoning_model,
            "temporal_normalization": config.temporal_normalization_model,
            "sensitivity": None,
        },
        "pipeline_metadata": {
            "input": str(args.input),
            "formatted_episode": formatted_episode,
            "compiler_attempt_count": compiler_result.attempt_count,
            "compiler_inference_duration_ms": compiler_result.inference_duration_ms,
            "assertion_count": len(compiler_result.output.assertions),
        },
        "assertions": assertion_outputs,
    }

    args.output.write_text(json.dumps(_to_jsonable(output), indent=2) + "\n", encoding="utf-8")
    print(f"Pipeline output written to {args.output}")
    return 0


def _run_semantic_compiler(episode, formatted_episode: str, config):
    started = time.perf_counter()
    client = OllamaClient(
        base_url=config.ollama_base_url,
        timeout_seconds=config.timeout_seconds,
    )
    raw_output = client.chat_structured(
        model=SEMANTIC_COMPILER_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_content=formatted_episode,
        json_schema=CompilerOutput.model_json_schema(),
        temperature=config.temperature,
        think=SEMANTIC_COMPILER_THINK,
    )
    output = CompilerOutput.model_validate(raw_output)

    return SimpleNamespace(
        output=output,
        model_name=SEMANTIC_COMPILER_MODEL,
        inference_duration_ms=int((time.perf_counter() - started) * 1000),
        attempt_count=1,
    )


def _resolve_argument(
    argument: SemanticArgument,
    assertion: CandidateSemanticAssertion,
    source_episode_id: str,
    errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    return _resolve_entity(
        text=argument.text,
        entity_type=argument.entity_type,
        source_episode_id=source_episode_id,
        role=argument.role,
        errors=errors,
    )


def _resolve_entity(
    *,
    text: str,
    entity_type: str | None,
    source_episode_id: str,
    role: str,
    errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    try:
        if entity_type is None:
            return _to_jsonable(resolve_entity_mention(text, entity_type))
        return _to_jsonable(resolve_or_create_entity(text, entity_type, source_episode_id))
    except Exception as exc:
        errors.append(
            {
                "stage": "entity_resolution",
                "role": role,
                "mention": text,
                "error": str(exc),
            }
        )
        return None


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
