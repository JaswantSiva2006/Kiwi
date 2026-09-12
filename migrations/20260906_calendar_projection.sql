CREATE TABLE IF NOT EXISTS calendar_events (
    calendar_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    semantic_memory_id UUID NOT NULL UNIQUE
        REFERENCES semantic_memories(memory_id)
        ON DELETE CASCADE,
    title TEXT NOT NULL,
    event_kind TEXT NULL,
    start_at TIMESTAMPTZ NULL,
    end_at TIMESTAMPTZ NULL,
    start_date DATE NULL,
    end_date DATE NULL,
    all_day BOOLEAN NOT NULL DEFAULT FALSE,
    location_text TEXT NULL,
    timezone_text TEXT NULL,
    recurrence TEXT NULL,
    recurrence_specifics TEXT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_calendar_events_status_start_at
ON calendar_events (status, start_at);

CREATE INDEX IF NOT EXISTS idx_calendar_events_status_start_date
ON calendar_events (status, start_date);

CREATE TABLE IF NOT EXISTS calendar_event_entities (
    calendar_event_id UUID NOT NULL
        REFERENCES calendar_events(calendar_event_id)
        ON DELETE CASCADE,
    entity_id UUID NOT NULL
        REFERENCES entities(entity_id),
    role TEXT NOT NULL,
    mention_text TEXT NULL,
    PRIMARY KEY (calendar_event_id, entity_id, role)
);

CREATE INDEX IF NOT EXISTS idx_calendar_event_entities_entity_id
ON calendar_event_entities (entity_id);
