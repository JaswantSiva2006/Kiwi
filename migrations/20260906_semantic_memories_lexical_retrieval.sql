CREATE INDEX IF NOT EXISTS idx_semantic_memories_canonical_text_fts
ON public.semantic_memories
USING gin (to_tsvector('simple', canonical_text));

CREATE INDEX IF NOT EXISTS idx_semantic_memories_canonical_text_trgm
ON public.semantic_memories
USING gin (canonical_text gin_trgm_ops);
