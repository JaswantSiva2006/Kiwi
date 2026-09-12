"""Local sentence-transformer embedding generation."""

from __future__ import annotations

from functools import lru_cache

from kivi_memory.embeddings.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = _model().encode(
        texts,
        batch_size=len(texts) or 1,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [[float(value) for value in embedding] for embedding in embeddings]
