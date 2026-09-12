from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from kivi_memory.common.config import KiviOrchestratorConfig
from kivi_memory.read_orchestrator import READ_ROUTER_THINK, ReadSideOrchestrator
from kivi_memory.read_orchestrator.models import ReadToolDecision
from kivi_memory.read_orchestrator.prompt import build_read_router_system_prompt
from kivi_memory.read_orchestrator.registry import ToolRegistry
from kivi_memory.working_memory.models import ThreadMessage


class FakeOllamaClient:
    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = list(outputs)
        self.calls = []

    def _post(self, path, payload):
        self.calls.append((path, payload))
        output = self.outputs.pop(0)
        return {
            "message": {"content": json.dumps(output)},
            "prompt_eval_count": 11,
            "eval_count": 7,
        }


@pytest.mark.parametrize(
    ("payload", "tools"),
    [
        ({"tool_calls": []}, []),
        ({"tool_calls": [{"tool": "semantic_memory.search", "arguments": {}}]}, ["semantic_memory.search"]),
        (
            {
                "tool_calls": [
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-15T00:00:00+05:30",
                            "end": "2026-09-16T00:00:00+05:30",
                        },
                    }
                ]
            },
            ["calendar.get_schedule"],
        ),
        (
            {"tool_calls": [{"tool": "redis_thread_history.search", "arguments": {"query": "optimization", "limit": 8}}]},
            ["redis_thread_history.search"],
        ),
        (
            {"tool_calls": [{"tool": "web.search", "arguments": {"query": "latest NVIDIA news", "freshness": "day"}}]},
            ["web.search"],
        ),
        (
            {
                "tool_calls": [
                    {"tool": "semantic_memory.search", "arguments": {}},
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-15T00:00:00+05:30",
                            "end": "2026-09-16T00:00:00+05:30",
                        },
                    },
                ]
            },
            ["semantic_memory.search", "calendar.get_schedule"],
        ),
        (
            {
                "tool_calls": [
                    {"tool": "semantic_memory.search", "arguments": {}},
                    {"tool": "redis_thread_history.search", "arguments": {"query": "old discussion"}},
                ]
            },
            ["semantic_memory.search", "redis_thread_history.search"],
        ),
        (
            {
                "tool_calls": [
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-15T00:00:00+05:30",
                            "end": "2026-09-16T00:00:00+05:30",
                        },
                    },
                    {"tool": "redis_thread_history.search", "arguments": {"query": "old discussion"}},
                ]
            },
            ["calendar.get_schedule", "redis_thread_history.search"],
        ),
        (
            {
                "tool_calls": [
                    {"tool": "semantic_memory.search", "arguments": {}},
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-15T00:00:00+05:30",
                            "end": "2026-09-16T00:00:00+05:30",
                        },
                    },
                    {"tool": "redis_thread_history.search", "arguments": {"query": "old discussion"}},
                ]
            },
            ["semantic_memory.search", "calendar.get_schedule", "redis_thread_history.search"],
        ),
        (
            {
                "tool_calls": [
                    {"tool": "semantic_memory.search", "arguments": {}},
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-15T00:00:00+05:30",
                            "end": "2026-09-16T00:00:00+05:30",
                        },
                    },
                    {"tool": "redis_thread_history.search", "arguments": {"query": "old discussion"}},
                    {"tool": "web.search", "arguments": {"query": "events next week", "freshness": "week"}},
                ]
            },
            ["semantic_memory.search", "calendar.get_schedule", "redis_thread_history.search", "web.search"],
        ),
    ],
)
def test_router_accepts_all_selection_shapes(payload, tools) -> None:
    client = FakeOllamaClient([payload])
    result = router(client).route(
        "query",
        [message("assistant", "Recent context.")],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert [call.tool for call in result.decision.tool_calls] == tools
    assert result.router_model_call_count == 1
    assert client.calls[0][1]["model"] == "qwen3.5:4b"
    assert client.calls[0][1]["think"] is READ_ROUTER_THINK is False
    assert client.calls[0][1]["options"] == {"temperature": 0.0}
    assert "semantic_memory.search" in client.calls[0][1]["messages"][0]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_calls": [{"tool": "unknown.tool", "arguments": {}}]},
        {"tool_calls": [{"tool": "web.search", "arguments": {"query": "", "freshness": "week"}}]},
        {"tool_calls": [{"tool": "web.search", "arguments": {"query": "NVIDIA", "freshness": "hour"}}]},
        {
            "tool_calls": [
                {"tool": "semantic_memory.search", "arguments": {}},
                {"tool": "semantic_memory.search", "arguments": {}},
            ]
        },
        {"tool_calls": [{"tool": "calendar.get_schedule", "arguments": {"start": "2026-09-15T00:00:00+05:30"}}]},
        {
            "tool_calls": [
                {
                    "tool": "calendar.get_schedule",
                    "arguments": {
                        "start": "2026-09-16T00:00:00+05:30",
                        "end": "2026-09-15T00:00:00+05:30",
                    },
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "tool": "redis_thread_history.search",
                    "arguments": {
                        "query": "optimization",
                        "start": "2026-09-11T10:00:00+05:30",
                        "end": "2026-09-11T10:00:00+05:30",
                    },
                }
            ]
        },
    ],
)
def test_invalid_tool_contract_retries_then_falls_back_to_no_tools(payload) -> None:
    client = FakeOllamaClient([payload, payload])
    result = router(client).route(
        "What do I have next Tuesday?",
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert result.fallback_used is True
    assert result.router_model_call_count == 2
    assert result.decision == ReadToolDecision(tool_calls=[])


def test_calendar_end_of_day_is_normalized_to_half_open_next_day() -> None:
    client = FakeOllamaClient(
        [
            {
                "tool_calls": [
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-15T00:00:00+05:30",
                            "end": "2026-09-15T23:59:59+05:30",
                        },
                    }
                ]
            }
        ]
    )
    result = router(client).route(
        "What do I have next Tuesday?",
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert result.fallback_used is False
    assert result.decision.tool_calls[0].arguments["end"] == "2026-09-16T00:00:00+05:30"


def test_explicit_next_weekday_calendar_date_is_repaired() -> None:
    client = FakeOllamaClient(
        [
            {
                "tool_calls": [
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-16T00:00:00+05:30",
                            "end": "2026-09-17T00:00:00+05:30",
                        },
                    }
                ]
            }
        ]
    )
    result = router(client).route(
        "What do I have next Tuesday?",
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert result.decision.tool_calls[0].arguments["start"] == "2026-09-15T00:00:00+05:30"
    assert result.decision.tool_calls[0].arguments["end"] == "2026-09-16T00:00:00+05:30"


def test_current_web_query_repairs_none_freshness() -> None:
    client = FakeOllamaClient(
        [
            {
                "tool_calls": [
                    {
                        "tool": "web.search",
                        "arguments": {"query": "latest NVIDIA news", "freshness": "none"},
                    }
                ]
            }
        ]
    )
    result = router(client).route(
        "What's the latest NVIDIA news?",
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert result.decision.tool_calls[0].arguments["freshness"] == "week"


def test_unusual_freshness_word_is_normalized() -> None:
    client = FakeOllamaClient(
        [
            {
                "tool_calls": [
                    {
                        "tool": "web.search",
                        "arguments": {"query": "current AI opportunities 2026", "freshness": "high"},
                    }
                ]
            }
        ]
    )
    result = router(client).route(
        "Find current AI opportunities that match my interests.",
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert result.fallback_used is False
    assert result.decision.tool_calls[0].arguments["freshness"] == "week"


def test_obvious_external_query_gets_web_if_router_omits_it() -> None:
    client = FakeOllamaClient([{"tool_calls": [{"tool": "semantic_memory.search", "arguments": {"query": "user interests"}}]}])
    result = router(client).route(
        "Find current AI opportunities that match my interests.",
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert [call.tool for call in result.decision.tool_calls] == ["semantic_memory.search", "web.search"]


@pytest.mark.parametrize(
    "query",
    [
        "What do you remember about Priya?",
        "Forget that I prefer morning meetings.",
        "Why did you think I prefer mornings?",
    ],
)
def test_obvious_memory_control_query_gets_memory_control_if_router_omits_it(query) -> None:
    client = FakeOllamaClient([{"tool_calls": []}])
    result = router(client).route(
        query,
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert [call.tool for call in result.decision.tool_calls] == ["memory.control"]
    assert result.decision.tool_calls[0].arguments["query"] == query


def test_explicit_next_week_calendar_range_is_repaired_to_calendar_week() -> None:
    client = FakeOllamaClient(
        [
            {
                "tool_calls": [
                    {
                        "tool": "calendar.get_schedule",
                        "arguments": {
                            "start": "2026-09-18T00:00:00+05:30",
                            "end": "2026-09-24T23:59:59+05:30",
                        },
                    }
                ]
            }
        ]
    )
    result = router(client).route(
        "Find events next week.",
        [],
        current_datetime=now(),
        user_timezone="Asia/Kolkata",
    )

    assert result.decision.tool_calls[0].arguments["start"] == "2026-09-14T00:00:00+05:30"
    assert result.decision.tool_calls[0].arguments["end"] == "2026-09-21T00:00:00+05:30"


def test_router_prompt_is_built_from_registry() -> None:
    prompt = build_read_router_system_prompt("- custom.tool: custom description")
    assert "custom.tool" in prompt
    assert "Tools are not mutually exclusive" in prompt


def router(client: FakeOllamaClient) -> ReadSideOrchestrator:
    return ReadSideOrchestrator(
        client=client,
        config=KiviOrchestratorConfig(router_model="qwen3.5:4b", router_temperature=0.0),
        registry=ToolRegistry(),
    )


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def message(role: str, text: str) -> ThreadMessage:
    return ThreadMessage(message_id="m1", role=role, text=text, timestamp=now().isoformat())
