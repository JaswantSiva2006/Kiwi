from __future__ import annotations

import os

import pytest

from kivi_memory.semantic_compiler.compiler import SemanticCompiler
from kivi_memory.common.schemas import MemoryEpisode


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("KIVI_RUN_OLLAMA_INTEGRATION") != "1",
    reason="set KIVI_RUN_OLLAMA_INTEGRATION=1 to call local Ollama",
)
def test_local_qwen_semantic_compiler_integration() -> None:
    episode = MemoryEpisode.model_validate(
        {
            "episode_id": "integration_1",
            "session_id": "test",
            "timezone": "Asia/Kolkata",
            "locale": "en-IN",
            "messages": [
                {
                    "message_id": "m1",
                    "role": "USER",
                    "timestamp": "2026-09-03T09:00:00",
                    "text": "Priya is handling Atlas now.",
                }
            ],
            "tool_context": None,
            "grounding_context": None,
        }
    )

    result = SemanticCompiler().compile(episode)

    assert result.validation_report.valid
    assert len(result.output.assertions) >= 1
