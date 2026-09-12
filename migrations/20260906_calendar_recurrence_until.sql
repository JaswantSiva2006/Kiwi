ALTER TABLE calendar_events
ADD COLUMN IF NOT EXISTS recurrence_until_at TIMESTAMPTZ NULL,
ADD COLUMN IF NOT EXISTS recurrence_until_date DATE NULL;

CREATE INDEX IF NOT EXISTS idx_calendar_events_status_recurrence_until_at
ON calendar_events (status, recurrence_until_at);

CREATE INDEX IF NOT EXISTS idx_calendar_events_status_recurrence_until_date
ON calendar_events (status, recurrence_until_date);
