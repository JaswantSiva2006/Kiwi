"""Configuration defaults and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TIMEZONE = "Asia/Kolkata"
DEFAULT_LOCALE = "en-IN"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_KIVI_MODEL = "qwen3.5:9b"
DEFAULT_KIVI_TEMPORAL_MODEL = "qwen3.5:9b"
DEFAULT_KIVI_TEMPORAL_REASONING_MODEL = "qwen3.5:9b"
DEFAULT_KIVI_TEMPORAL_NORMALIZATION_MODEL = "qwen3.5:9b"
DEFAULT_KIVI_TEMPERATURE = 0.0
DEFAULT_KIVI_MAX_ATTEMPTS = 2
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
DEFAULT_KIVI_THINK = True
PIPELINE_SEMANTIC_COMPILER_MODEL = "qwen3.5:9b"
PIPELINE_SEMANTIC_COMPILER_THINK = True
PIPELINE_FINAL_TOP_K = 8
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_KIVI_THREAD_TTL_SECONDS = 604800
DEFAULT_KIVI_THREAD_MAX_MESSAGES = 500
DEFAULT_KIVI_THREAD_CONTEXT_MESSAGES = 24
DEFAULT_KIVI_THREAD_CONTEXT_MAX_CHARS = 12000
DEFAULT_KIVI_THREAD_CONTEXT_EPISODES = 6
DEFAULT_KIVI_THREAD_CONTEXT_MAX_TOKENS = 2500
DEFAULT_KIVI_THREAD_ACTIVE_WINDOW_EPISODES = 20
DEFAULT_KIVI_THREAD_ACTIVE_WINDOW_TOKENS = 8000
DEFAULT_KIVI_WRITEBACK_BATCH_SIZE = 3
DEFAULT_KIVI_WRITEBACK_IDLE_FLUSH_SECONDS = 180
DEFAULT_KIVI_WRITEBACK_ON_TURN_ENABLED = True
DEFAULT_KIVI_WRITEBACK_LOCK_SECONDS = 300
DEFAULT_KIVI_ROUTER_MODEL = "qwen3.5:4b"
DEFAULT_KIVI_ROUTER_TEMPERATURE = 0.0
DEFAULT_KIVI_ROUTER_MAX_RETRIEVAL_QUERIES = 3
DEFAULT_KIVI_ORCHESTRATOR_MEMORY_LIMIT = 16
DEFAULT_KIVI_HEY_KIVI_PROVIDER = "sarvam"
DEFAULT_KIVI_HEY_KIVI_MODEL = "sarvam-105b"
DEFAULT_KIVI_HEY_KIVI_REASONING_EFFORT = "low"
DEFAULT_KIVI_HEY_KIVI_TEMPERATURE = 0.2
DEFAULT_KIVI_HEY_KIVI_MAX_TOKENS = 4096
DEFAULT_KIVI_HEY_KIVI_TIMEOUT_SECONDS = 45.0
DEFAULT_KIVI_AGENT_MAX_MEMORIES = 12


@dataclass(frozen=True)
class KiviThreadMemoryConfig:
    redis_url: str = DEFAULT_REDIS_URL
    ttl_seconds: int = DEFAULT_KIVI_THREAD_TTL_SECONDS
    max_messages: int = DEFAULT_KIVI_THREAD_MAX_MESSAGES
    context_messages: int = DEFAULT_KIVI_THREAD_CONTEXT_MESSAGES
    context_max_chars: int = DEFAULT_KIVI_THREAD_CONTEXT_MAX_CHARS
    context_episodes: int = DEFAULT_KIVI_THREAD_CONTEXT_EPISODES
    context_max_tokens: int = DEFAULT_KIVI_THREAD_CONTEXT_MAX_TOKENS
    active_window_episodes: int = DEFAULT_KIVI_THREAD_ACTIVE_WINDOW_EPISODES
    active_window_tokens: int = DEFAULT_KIVI_THREAD_ACTIVE_WINDOW_TOKENS
    writeback_batch_size: int = DEFAULT_KIVI_WRITEBACK_BATCH_SIZE
    idle_flush_after_seconds: int = DEFAULT_KIVI_WRITEBACK_IDLE_FLUSH_SECONDS
    writeback_on_turn_enabled: bool = DEFAULT_KIVI_WRITEBACK_ON_TURN_ENABLED
    writeback_lock_seconds: int = DEFAULT_KIVI_WRITEBACK_LOCK_SECONDS


@dataclass(frozen=True)
class KiviOrchestratorConfig:
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    router_model: str = DEFAULT_KIVI_ROUTER_MODEL
    router_temperature: float = DEFAULT_KIVI_ROUTER_TEMPERATURE
    router_max_retrieval_queries: int = DEFAULT_KIVI_ROUTER_MAX_RETRIEVAL_QUERIES
    memory_limit: int = DEFAULT_KIVI_ORCHESTRATOR_MEMORY_LIMIT
    hey_kivi_provider: str = DEFAULT_KIVI_HEY_KIVI_PROVIDER
    hey_kivi_model: str = DEFAULT_KIVI_HEY_KIVI_MODEL
    sarvam_api_key: str | None = None
    hey_kivi_reasoning_effort: str = DEFAULT_KIVI_HEY_KIVI_REASONING_EFFORT
    hey_kivi_temperature: float = DEFAULT_KIVI_HEY_KIVI_TEMPERATURE
    hey_kivi_max_tokens: int = DEFAULT_KIVI_HEY_KIVI_MAX_TOKENS
    hey_kivi_timeout_seconds: float = DEFAULT_KIVI_HEY_KIVI_TIMEOUT_SECONDS
    agent_max_memories: int = DEFAULT_KIVI_AGENT_MAX_MEMORIES


@dataclass(frozen=True)
class KiviCompilerConfig:
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_KIVI_MODEL
    temporal_model: str = DEFAULT_KIVI_TEMPORAL_MODEL
    temporal_reasoning_model: str = DEFAULT_KIVI_TEMPORAL_REASONING_MODEL
    temporal_normalization_model: str = DEFAULT_KIVI_TEMPORAL_NORMALIZATION_MODEL
    temperature: float = DEFAULT_KIVI_TEMPERATURE
    max_attempts: int = DEFAULT_KIVI_MAX_ATTEMPTS
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    think: bool | None = DEFAULT_KIVI_THINK


def load_config_from_env() -> KiviCompilerConfig:
    """Load compiler runtime config from environment variables."""

    return KiviCompilerConfig(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        model=os.getenv("KIVI_SEMANTIC_COMPILER_MODEL", os.getenv("KIVI_MODEL", DEFAULT_KIVI_MODEL)),
        temporal_model=os.getenv("KIVI_TEMPORAL_MODEL", DEFAULT_KIVI_TEMPORAL_MODEL),
        temporal_reasoning_model=os.getenv(
            "KIVI_TEMPORAL_REASONING_MODEL",
            os.getenv("KIVI_TEMPORAL_MODEL", DEFAULT_KIVI_TEMPORAL_REASONING_MODEL),
        ),
        temporal_normalization_model=os.getenv(
            "KIVI_TEMPORAL_NORMALIZATION_MODEL",
            DEFAULT_KIVI_TEMPORAL_NORMALIZATION_MODEL,
        ),
        temperature=float(os.getenv("KIVI_TEMPERATURE", str(DEFAULT_KIVI_TEMPERATURE))),
        max_attempts=int(os.getenv("KIVI_MAX_ATTEMPTS", str(DEFAULT_KIVI_MAX_ATTEMPTS))),
        timeout_seconds=float(
            os.getenv("KIVI_OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
        ),
        think=_optional_bool_env(
            "KIVI_SEMANTIC_COMPILER_THINK",
            _parse_optional_bool(os.getenv("KIVI_THINK")),
            DEFAULT_KIVI_THINK,
        ),
    )


def load_thread_memory_config_from_env() -> KiviThreadMemoryConfig:
    """Load Redis working-memory settings from environment variables."""

    return KiviThreadMemoryConfig(
        redis_url=os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
        ttl_seconds=_positive_int_env("KIVI_THREAD_TTL_SECONDS", DEFAULT_KIVI_THREAD_TTL_SECONDS),
        max_messages=_positive_int_env("KIVI_THREAD_MAX_MESSAGES", DEFAULT_KIVI_THREAD_MAX_MESSAGES),
        context_messages=_positive_int_env("KIVI_THREAD_CONTEXT_MESSAGES", DEFAULT_KIVI_THREAD_CONTEXT_MESSAGES),
        context_max_chars=_positive_int_env("KIVI_THREAD_CONTEXT_MAX_CHARS", DEFAULT_KIVI_THREAD_CONTEXT_MAX_CHARS),
        context_episodes=_positive_int_env("KIVI_THREAD_CONTEXT_EPISODES", DEFAULT_KIVI_THREAD_CONTEXT_EPISODES),
        context_max_tokens=_positive_int_env("KIVI_THREAD_CONTEXT_MAX_TOKENS", DEFAULT_KIVI_THREAD_CONTEXT_MAX_TOKENS),
        active_window_episodes=_positive_int_env(
            "KIVI_THREAD_ACTIVE_WINDOW_EPISODES",
            DEFAULT_KIVI_THREAD_ACTIVE_WINDOW_EPISODES,
        ),
        active_window_tokens=_positive_int_env(
            "KIVI_THREAD_ACTIVE_WINDOW_TOKENS",
            DEFAULT_KIVI_THREAD_ACTIVE_WINDOW_TOKENS,
        ),
        writeback_batch_size=_positive_int_env("KIVI_WRITEBACK_BATCH_SIZE", DEFAULT_KIVI_WRITEBACK_BATCH_SIZE),
        idle_flush_after_seconds=_positive_int_env(
            "KIVI_WRITEBACK_IDLE_FLUSH_SECONDS",
            DEFAULT_KIVI_WRITEBACK_IDLE_FLUSH_SECONDS,
        ),
        writeback_on_turn_enabled=_parse_bool_env(
            "KIVI_WRITEBACK_ON_TURN_ENABLED",
            DEFAULT_KIVI_WRITEBACK_ON_TURN_ENABLED,
        ),
        writeback_lock_seconds=_positive_int_env("KIVI_WRITEBACK_LOCK_SECONDS", DEFAULT_KIVI_WRITEBACK_LOCK_SECONDS),
    )


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_orchestrator_config_from_env() -> KiviOrchestratorConfig:
    """Load Hey Kivi orchestration settings from environment variables."""

    return KiviOrchestratorConfig(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        timeout_seconds=float(os.getenv("KIVI_OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))),
        router_model=os.getenv("KIVI_ROUTER_MODEL", DEFAULT_KIVI_ROUTER_MODEL),
        router_temperature=float(os.getenv("KIVI_ROUTER_TEMPERATURE", str(DEFAULT_KIVI_ROUTER_TEMPERATURE))),
        router_max_retrieval_queries=_bounded_positive_int_env(
            "KIVI_ROUTER_MAX_RETRIEVAL_QUERIES",
            DEFAULT_KIVI_ROUTER_MAX_RETRIEVAL_QUERIES,
            3,
        ),
        memory_limit=_positive_int_env("KIVI_ORCHESTRATOR_MEMORY_LIMIT", DEFAULT_KIVI_ORCHESTRATOR_MEMORY_LIMIT),
        hey_kivi_provider=os.getenv("KIVI_HEY_KIVI_PROVIDER", DEFAULT_KIVI_HEY_KIVI_PROVIDER),
        hey_kivi_model=os.getenv("KIVI_HEY_KIVI_MODEL", DEFAULT_KIVI_HEY_KIVI_MODEL),
        sarvam_api_key=_optional_non_empty_env("SARVAM_API_KEY"),
        hey_kivi_reasoning_effort=os.getenv(
            "KIVI_HEY_KIVI_REASONING_EFFORT",
            DEFAULT_KIVI_HEY_KIVI_REASONING_EFFORT,
        ),
        hey_kivi_temperature=float(os.getenv("KIVI_HEY_KIVI_TEMPERATURE", str(DEFAULT_KIVI_HEY_KIVI_TEMPERATURE))),
        hey_kivi_max_tokens=_positive_int_env("KIVI_HEY_KIVI_MAX_TOKENS", DEFAULT_KIVI_HEY_KIVI_MAX_TOKENS),
        hey_kivi_timeout_seconds=float(
            os.getenv("KIVI_HEY_KIVI_TIMEOUT_SECONDS", str(DEFAULT_KIVI_HEY_KIVI_TIMEOUT_SECONDS))
        ),
        agent_max_memories=_positive_int_env("KIVI_AGENT_MAX_MEMORIES", DEFAULT_KIVI_AGENT_MAX_MEMORIES),
    )


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_bool_env(name: str, fallback: bool | None, default: bool) -> bool:
    parsed = _parse_optional_bool(os.getenv(name))
    if parsed is not None:
        return parsed
    if fallback is not None:
        return fallback
    return default


def _optional_non_empty_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_positive_int_env(name: str, default: int, maximum: int) -> int:
    return min(_positive_int_env(name, default), maximum)
