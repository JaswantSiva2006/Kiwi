"""Conservative entity-name normalization."""

from __future__ import annotations

import re
import unicodedata

_SIMPLE_PUNCTUATION = str.maketrans({char: " " for char in "\"'.,;:!?()[]{}<>"})
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_entity_name(text: str) -> str:
    """Normalize an entity mention or stored entity label for matching."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = normalized.translate(_SIMPLE_PUNCTUATION)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()
