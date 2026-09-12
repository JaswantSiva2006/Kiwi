"""Shared configuration defaults for entity resolution."""

from __future__ import annotations

import os

DEFAULT_DATABASE_URL = "postgresql://kivi:kivi_dev@localhost:5432/kivi_memory"
ENTITY_FUZZY_CANDIDATE_THRESHOLD = 0.30
ENTITY_PLAUSIBLE_MATCH_THRESHOLD = 0.50
ENTITY_AUTO_MATCH_THRESHOLD = 0.85
ENTITY_MIN_SCORE_MARGIN = 0.10


def get_database_url() -> str:
    """Return the configured Kivi memory database URL."""

    return os.getenv("KIVI_DATABASE_URL", DEFAULT_DATABASE_URL)
