from __future__ import annotations

from kivi_memory.common.schemas import CompilerOutput, MemoryEpisode, TemporalMetadata, ValidationReport
from kivi_memory.pipeline.write_pipeline import WritePipeline
from kivi_memory.semantic_compiler.compiler import CompilerResult
from tests.common.test_input_schema import valid_episode_data
from tests.enrichment.test_validator import base_assertion


class MockSemanticCompiler:
    def compile(self, episode: MemoryEpisode) -> CompilerResult:
        output = CompilerOutput(assertions=[base_assertion()])
        return CompilerResult(
            output=output,
            validation_report=ValidationReport(valid=True, issues=[]),
            model_name="mock",
            inference_duration_ms=1,
            attempt_count=1,
            diagnostics={},
        )


class MockTemporalNormalizer:
    def normalize(self, assertion, episode):
        class Result:
            temporal_metadata = TemporalMetadata(
                temporal_kind="NONE",
                valid_from_hint=None,
                valid_to_hint=None,
                event_time=None,
                temporal_precision="NONE",
                recurrence="NONE",
                recurrence_specifics=None,
            )

        return Result()


def test_write_pipeline_uses_compiler_and_validator() -> None:
    episode = MemoryEpisode.model_validate(valid_episode_data())
    result = WritePipeline(
        semantic_compiler=MockSemanticCompiler(),
        temporal_normalizer=MockTemporalNormalizer(),
    ).process(episode)

    assert result.episode_id == episode.episode_id
    assert len(result.compiler_result.output.assertions) == 1
    assert len(result.enriched_assertions) == 1
    assert result.enriched_assertions[0].validation_report.valid
    assert result.enriched_assertions[0].temporal_metadata is not None
