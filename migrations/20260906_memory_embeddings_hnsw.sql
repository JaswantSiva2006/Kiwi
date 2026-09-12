CREATE INDEX CONCURRENTLY IF NOT EXISTS memory_embeddings_minilm_hnsw_cosine_idx
ON public.memory_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
)
WHERE embedding_model = 'sentence-transformers/all-MiniLM-L6-v2';
