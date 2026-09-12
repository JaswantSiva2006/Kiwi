from __future__ import annotations

import math

from kivi_memory.embeddings import EMBEDDING_DIMENSION, embed_text


def test_embedding_dimension_is_384() -> None:
    embedding = embed_text("The user met Kavya yesterday.")

    assert len(embedding) == EMBEDDING_DIMENSION


def test_embedding_is_normalized() -> None:
    embedding = embed_text("The user met Kavya yesterday.")

    norm = math.sqrt(sum(value * value for value in embedding))
    assert norm == pytest_approx(1.0)


def test_same_text_produces_near_identical_embedding() -> None:
    first = embed_text("The user met Kavya yesterday.")
    second = embed_text("The user met Kavya yesterday.")

    max_delta = max(abs(left - right) for left, right in zip(first, second, strict=True))
    assert max_delta < 1e-6


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-5)
