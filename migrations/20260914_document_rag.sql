CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS document_rag;

CREATE TABLE IF NOT EXISTS document_rag.documents (
    document_id text PRIMARY KEY,
    filename text NOT NULL,
    sha256 text NOT NULL UNIQUE,
    mime_type text NOT NULL DEFAULT 'application/pdf',
    page_count integer NOT NULL CHECK (page_count >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL CHECK (status IN ('EMBEDDING', 'READY', 'FAILED'))
);

CREATE TABLE IF NOT EXISTS document_rag.chunks (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES document_rag.documents(document_id) ON DELETE CASCADE,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    text text NOT NULL CHECK (length(trim(text)) > 0),
    section_title text,
    section_path jsonb NOT NULL DEFAULT '[]'::jsonb,
    page_start integer NOT NULL CHECK (page_start >= 1),
    page_end integer NOT NULL CHECK (page_end >= 1),
    token_count integer NOT NULL CHECK (token_count > 0),
    embedding vector(384) NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (document_id, chunk_index),
    CHECK (page_start <= page_end)
);

CREATE INDEX IF NOT EXISTS document_rag_chunks_document_id_idx
ON document_rag.chunks (document_id);

CREATE INDEX IF NOT EXISTS document_rag_chunks_hnsw_cosine_idx
ON document_rag.chunks
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);
