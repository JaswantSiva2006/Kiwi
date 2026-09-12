"""Semantic compiler orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from kivi_memory.common.config import KiviCompilerConfig, load_config_from_env
from kivi_memory.ingestion.episode_formatter import format_memory_episode
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError
from kivi_memory.enrichment.validator import validate_compiler_output
from kivi_memory.semantic_compiler.prompt import RETRY_INSTRUCTION, SYSTEM_PROMPT
from kivi_memory.common.schemas import CompilerOutput, MemoryEpisode, ValidationReport

logger = logging.getLogger(__name__)


class SemanticCompilerError(RuntimeError):
    """Raised when local semantic compilation cannot produce valid output."""


@dataclass(frozen=True)
class CompilerResult:
    output: CompilerOutput
    validation_report: ValidationReport
    model_name: str
    inference_duration_ms: int
    attempt_count: int
    diagnostics: dict[str, Any]


class SemanticCompiler:
    def __init__(
        self,
        client: OllamaClient | None = None,
        config: KiviCompilerConfig | None = None,
    ) -> None:
        self.config = config or load_config_from_env()
        self.client = client or OllamaClient(
            base_url=self.config.ollama_base_url,
            timeout_seconds=self.config.timeout_seconds,
        )

    def compile(self, episode: MemoryEpisode) -> CompilerResult:
        formatted_episode = format_memory_episode(episode)
        schema = compiler_output_json_schema()
        started = time.perf_counter()
        last_error: Exception | None = None

        logger.info("Starting semantic compile episode=%s model=%s", episode.episode_id, self.config.model)

        for attempt in range(1, self.config.max_attempts + 1):
            user_content = formatted_episode if attempt == 1 else f"{formatted_episode}\n\n{RETRY_INSTRUCTION}"
            if attempt > 1:
                logger.warning("Retrying semantic compile episode=%s attempt=%s", episode.episode_id, attempt)

            try:
                raw = self.client.chat_structured(
                    model=self.config.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_content=user_content,
                    json_schema=schema,
                    temperature=self.config.temperature,
                    think=self.config.think,
                )
                output = CompilerOutput.model_validate(_repair_source_span_message_ids(raw, episode))
                report = validate_compiler_output(output, episode)
                if report.valid:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    logger.info(
                        "Semantic compile complete episode=%s model=%s assertions=%s valid=%s duration_ms=%s",
                        episode.episode_id,
                        self.config.model,
                        len(output.assertions),
                        report.valid,
                        duration_ms,
                    )
                    return CompilerResult(
                        output=output,
                        validation_report=report,
                        model_name=self.config.model,
                        inference_duration_ms=duration_ms,
                        attempt_count=attempt,
                        diagnostics={"assertion_count": len(output.assertions)},
                    )

                last_error = SemanticCompilerError(_format_report_errors(report))
            except (OllamaClientError, ValidationError, ValueError) as exc:
                last_error = exc

        duration_ms = int((time.perf_counter() - started) * 1000)
        raise SemanticCompilerError(
            f"Semantic compiler failed after {self.config.max_attempts} attempt(s) "
            f"in {duration_ms} ms: {last_error}"
        )


def _format_report_errors(report: ValidationReport) -> str:
    errors = [issue for issue in report.issues if issue.severity == "ERROR"]
    return "; ".join(f"{issue.code}: {issue.message}" for issue in errors) or "invalid output"


def _repair_source_span_message_ids(raw: dict[str, Any], episode: MemoryEpisode) -> dict[str, Any]:
    """Map model-invented source IDs to the unique episode message containing the span."""

    assertions = raw.get("assertions")
    if not isinstance(assertions, list):
        return raw
    known_ids = {message.message_id for message in episode.messages}
    repaired = dict(raw)
    repaired_assertions = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            repaired_assertions.append(assertion)
            continue
        assertion = dict(assertion)
        spans = assertion.get("source_spans")
        if isinstance(spans, list):
            repaired_spans = []
            for span in spans:
                if isinstance(span, dict):
                    span = dict(span)
                    if span.get("message_id") not in known_ids:
                        replacement = _unique_message_id_for_span(str(span.get("text", "")), episode)
                        if replacement is not None:
                            span["message_id"] = replacement
                repaired_spans.append(span)
            assertion["source_spans"] = repaired_spans
        repaired_assertions.append(assertion)
    repaired["assertions"] = repaired_assertions
    return repaired


def _unique_message_id_for_span(text: str, episode: MemoryEpisode) -> str | None:
    if not text:
        return None
    matches = [message.message_id for message in episode.messages if text in message.text]
    return matches[0] if len(matches) == 1 else None


def compiler_output_json_schema() -> dict[str, Any]:
    """Return the structured-output schema used for semantic-only generation."""

    return CompilerOutput.model_json_schema()
