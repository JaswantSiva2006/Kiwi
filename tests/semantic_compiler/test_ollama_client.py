from __future__ import annotations

import json

import httpx
import pytest

from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaResponseError


def test_chat_structured_forms_expected_payload() -> None:
    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"assertions":[]}'}})

    client = _mocked_client(handler)
    result = client.chat_structured(
        model="qwen-test",
        system_prompt="system",
        user_content="episode",
        json_schema={"type": "object"},
        temperature=0,
    )

    assert result == {"assertions": []}
    assert seen_payload["model"] == "qwen-test"
    assert seen_payload["stream"] is False
    assert seen_payload["format"] == {"type": "object"}
    assert seen_payload["options"]["temperature"] == 0
    assert seen_payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "episode"},
    ]


def test_chat_structured_rejects_malformed_generated_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not-json"}})

    client = _mocked_client(handler)
    with pytest.raises(OllamaResponseError):
        client.chat_structured("qwen-test", "system", "episode", {"type": "object"})


def test_list_models_parses_local_model_names() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen-test"}, {"name": "other"}]})

    client = _mocked_client(handler)
    assert client.list_models() == ["qwen-test", "other"]
    assert client.has_model("qwen-test")


def _mocked_client(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    return OllamaClient("http://testserver", client_factory=lambda **kwargs: httpx.Client(transport=transport))
