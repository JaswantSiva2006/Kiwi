"""Lightweight local Ollama API client."""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaClientError(RuntimeError):
    """Base error for local Ollama communication."""


class OllamaUnavailableError(OllamaClientError):
    """Ollama endpoint could not be reached."""


class OllamaModelUnavailableError(OllamaClientError):
    """The configured model is not locally available."""


class OllamaRequestTimeoutError(OllamaClientError):
    """The local Ollama request timed out."""


class OllamaResponseError(OllamaClientError):
    """Ollama returned an unusable response."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        keep_alive: str = "24h",
        client_factory=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self._client_factory = client_factory or httpx.Client

    def chat_structured(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        json_schema: dict[str, Any],
        temperature: float = 0.0,
        think: bool | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": json_schema,
            "options": {"temperature": temperature},
            "keep_alive": self.keep_alive,
        }
        if think is not None:
            payload["think"] = think
        response = self._post("/api/chat", payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama response did not contain message.content")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Ollama returned malformed generated JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise OllamaResponseError("Ollama generated JSON was not an object")
        return parsed

    def list_models(self) -> list[str]:
        response = self._get("/api/tags")
        models = response.get("models", [])
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama /api/tags response did not contain models list")
        return [model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)]

    def has_model(self, model_name: str) -> bool:
        return model_name in self.list_models()

    def health(self) -> bool:
        self._get("/api/tags")
        return True

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with self._client_factory(timeout=self.timeout_seconds) as client:
                response = client.get(f"{self.base_url}{path}")
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(f"Ollama is unavailable at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise OllamaRequestTimeoutError("Ollama request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(f"Ollama returned HTTP {exc.response.status_code}") from exc
        return _response_json(response)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._client_factory(timeout=self.timeout_seconds) as client:
                response = client.post(f"{self.base_url}{path}", json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(f"Ollama is unavailable at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise OllamaRequestTimeoutError("Ollama request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 404:
                raise OllamaModelUnavailableError(f"Ollama model {payload.get('model')!r} is unavailable") from exc
            raise OllamaResponseError(f"Ollama returned HTTP {status}") from exc
        return _response_json(response)


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise OllamaResponseError("Ollama returned malformed API JSON") from exc
    if not isinstance(data, dict):
        raise OllamaResponseError("Ollama API response was not an object")
    return data
