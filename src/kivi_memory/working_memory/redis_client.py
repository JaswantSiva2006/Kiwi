"""Shared Redis client/pool for within-thread working memory."""

from __future__ import annotations

from functools import lru_cache

import redis

from kivi_memory.common.config import load_thread_memory_config_from_env


@lru_cache(maxsize=1)
def get_redis_pool(redis_url: str | None = None) -> redis.ConnectionPool:
    config = load_thread_memory_config_from_env()
    return redis.ConnectionPool.from_url(redis_url or config.redis_url, decode_responses=True)


@lru_cache(maxsize=1)
def get_redis_client(redis_url: str | None = None) -> redis.Redis:
    return redis.Redis(connection_pool=get_redis_pool(redis_url))
