from __future__ import annotations

from functools import lru_cache

from .config import DOCUMENT_EMBEDDING_DIMENSION, DOCUMENT_EMBEDDING_MODEL_NAME, get_document_embedding_batch_size


class DocumentEmbeddingService:
    """Batched embedding facade for Document RAG.

    The underlying SentenceTransformer model is cached by the shared embedding
    generator, so this service can be constructed freely without reloading it.
    """

    model_name = DOCUMENT_EMBEDDING_MODEL_NAME
    dimension = DOCUMENT_EMBEDDING_DIMENSION

    def __init__(self, batch_size: int | None = None) -> None:
        self.batch_size = batch_size or get_document_embedding_batch_size()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(_embed_texts(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DOCUMENT_EMBEDDING_MODEL_NAME)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = _model().encode(
        texts,
        batch_size=len(texts) or 1,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [[float(value) for value in embedding] for embedding in embeddings]
