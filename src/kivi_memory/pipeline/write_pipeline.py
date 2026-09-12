"""Current write pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from kivi_memory.common.schemas import CandidateSemanticAssertion, MemoryEpisode, TemporalMetadata, ValidationReport
from kivi_memory.enrichment.temporal import TemporalNormalizer
from kivi_memory.enrichment.validator import validate_assertions
from kivi_memory.semantic_compiler.compiler import CompilerResult, SemanticCompiler


@dataclass(frozen=True)
class EnrichedAssertionResult:
    assertion: CandidateSemanticAssertion
    validation_report: ValidationReport
    temporal_metadata: TemporalMetadata | None


@dataclass(frozen=True)
class WritePipelineResult:
    episode_id: str
    compiler_result: CompilerResult
    enriched_assertions: list[EnrichedAssertionResult]

    @property
    def validated_assertions(self) -> list[EnrichedAssertionResult]:
        return self.enriched_assertions


class WritePipeline:
    """Run the implemented write path without storage side effects."""

    def __init__(
        self,
        semantic_compiler: SemanticCompiler | None = None,
        temporal_normalizer: TemporalNormalizer | None = None,
    ) -> None:
        self.semantic_compiler = semantic_compiler or SemanticCompiler()
        self.temporal_normalizer = temporal_normalizer or TemporalNormalizer()

    def process(self, episode: MemoryEpisode) -> WritePipelineResult:
        compiler_result = self.semantic_compiler.compile(episode)
        validated_assertions = validate_assertions(compiler_result.output.assertions, episode)
        enriched_assertions: list[EnrichedAssertionResult] = []

        # Planned fan-out extension points:
        # - Sensitivity Classifier
        for result in validated_assertions:
            temporal_metadata = None
            if result.validation_report.valid:
                temporal_metadata = self.temporal_normalizer.normalize(result.assertion, episode).temporal_metadata
            enriched_assertions.append(
                EnrichedAssertionResult(
                    assertion=result.assertion,
                    validation_report=result.validation_report,
                    temporal_metadata=temporal_metadata,
                )
            )

        return WritePipelineResult(
            episode_id=episode.episode_id,
            compiler_result=compiler_result,
            enriched_assertions=enriched_assertions,
        )
