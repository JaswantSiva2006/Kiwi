CREATE TABLE IF NOT EXISTS public.memory_entity_links (
    memory_id UUID NOT NULL REFERENCES public.semantic_memories(memory_id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES public.entities(entity_id),
    link_type TEXT NOT NULL,
    argument_role TEXT NULL,
    predicate_type TEXT NOT NULL,
    position INTEGER NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT memory_entity_links_link_type_check
        CHECK (link_type IN ('SUBJECT', 'ARGUMENT')),
    CONSTRAINT memory_entity_links_shape_check
        CHECK (
            (link_type = 'SUBJECT' AND argument_role IS NULL AND position IS NULL)
            OR
            (link_type = 'ARGUMENT' AND argument_role IS NOT NULL AND position IS NOT NULL)
        )
);

CREATE UNIQUE INDEX IF NOT EXISTS memory_entity_links_subject_unique_idx
ON public.memory_entity_links (memory_id, entity_id, predicate_type)
WHERE link_type = 'SUBJECT';

CREATE UNIQUE INDEX IF NOT EXISTS memory_entity_links_argument_unique_idx
ON public.memory_entity_links (memory_id, entity_id, argument_role, predicate_type, position)
WHERE link_type = 'ARGUMENT';

CREATE INDEX IF NOT EXISTS idx_memory_entity_links_entity_id
ON public.memory_entity_links (entity_id);

CREATE INDEX IF NOT EXISTS idx_memory_entity_links_memory_id
ON public.memory_entity_links (memory_id);

CREATE INDEX IF NOT EXISTS idx_memory_entity_links_entity_predicate
ON public.memory_entity_links (entity_id, predicate_type);
