from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.common.config import load_config_from_env
from kivi_memory.semantic_compiler.ollama_client import OllamaClient, OllamaClientError


def main() -> int:
    config = load_config_from_env()
    client = OllamaClient(config.ollama_base_url, config.timeout_seconds)

    print(f"Ollama URL: {config.ollama_base_url}")
    print(f"Configured model: {config.model}")
    print(f"Configured temporal model: {config.temporal_model}")

    try:
        models = client.list_models()
    except OllamaClientError as exc:
        print(f"Ollama check failed: {exc}")
        return 1

    print(f"Reachable: yes")
    print(f"Local models: {len(models)}")
    if config.model not in models:
        print(f"Model available: no")
        print(f"Requested local model was not found: {config.model}")
        return 1
    if config.temporal_model not in models:
        print(f"Temporal model available: no")
        print(f"Requested local temporal model was not found: {config.temporal_model}")
        return 1

    print("Model available: yes")
    print("Temporal model available: yes")
    print("Health check: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
