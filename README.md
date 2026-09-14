# Kivi

Kivi is a personal AI interface with persistent semantic understanding across interactions. Durable information is extracted into structured semantic memories, resolved against entities, reconciled with prior knowledge, stored with provenance, and selectively retrieved for future answers and tools.

This repo is a local prototype with a FastAPI backend, Vite/React frontend, PostgreSQL semantic ledger, Redis short-term thread context, Ollama local models for memory work, and Sarvam for final answer generation.

## New Additions

- PDF and Doc Parsing + Semantic storage (Added) (Branch: With_doc_ingestion)
- New UI (Added) (Branch: With_doc_ingestion)

## Product Documents

- [Product Positioning](./product_positioning.md)
- [Product Vision](./product_vision.md)

## Core Capabilities

- Persistent semantic memory for facts, preferences, people, projects, decisions, routines, commitments, and calendar events.
- Atomic assertion extraction with source spans and provenance.
- Temporal interpretation, recurrence, and calendar metadata downstream of semantic extraction.
- Entity resolution with aliases and entity-memory links.
- Candidate retrieval across vector, lexical, structured, and graph branches.
- Reconciliation operations: `ADD`, `REINFORCE`, `SUPERSEDE`, `RETRACT`, `NO_MEMORY`.
- Canonical PostgreSQL ledger plus derived embedding, graph, and calendar projections.
- Redis-backed recent thread context and optional Redis-to-durable writeback.
- Read-side tool routing over semantic memory, calendar, Redis history, web search, and memory control.
- Reproducible corpus import, ingestion, inspection, reset, and evaluation scripts.

## Semantic Memory Vs Chat History

```text
Redis/current thread       recent visible USER/ASSISTANT turns for local context
Durable semantic memory    structured user knowledge in PostgreSQL
Derived projections        embeddings, graph links, and calendar rows for retrieval/tools
```

Recent Redis context helps with local references and continuity. Durable memory stores user knowledge that should survive thread boundaries. Kivi does not replay all historic text into prompts; it extracts durable assertions, records evidence, resolves entities, reconciles conflicts, and retrieves only relevant active memories later.

## Architecture

Write path:

```text
Dictation / conversation / corpus JSONL
        |
MemoryEpisode
        |
SemanticCompiler
        |
validate_assertions
        |
TemporalGate -> TemporalNormalizer
        |
EntityResolutionOrchestrator
        |
Candidate retrieval: vector / structured / graph
        |
Reconciler
ADD / REINFORCE / SUPERSEDE / RETRACT / NO_MEMORY
        |
Ledger mutation
        |
Post-commit projections: embedding / graph / calendar
```

Read path:

```text
User query
        |
Recent Redis ThreadEpisode context
        |
ReadSideOrchestrator tool router
        |
0..N tools
        |
Context builder
        |
Sarvam final-answer model
        |
Assistant answer
        |
ThreadEpisode appended to Redis
        |
optional background durable writeback
```

## Memory Lifecycle

An observation becomes memory through: source `MemoryEpisode`, atomic assertions, source spans, temporal processing, entity resolution, candidate retrieval, reconciliation, deterministic mutation, active ledger state, ledger events, projections, and future retrieval.

Example:

```text
"Rohit handles Atlas."
later:
"Priya handles Atlas now."
```

If the incoming assertion clearly updates the same current responsibility slot, reconciliation can choose `SUPERSEDE`, preserving the old memory historically and making the newer memory active.

```text
"Rohit never handled Atlas; that earlier information was wrong."
```

This is correction/invalidation and can choose `RETRACT`.

## Temporal And Calendar

The Semantic Compiler preserves semantic meaning and temporal wording but does not own detailed calendar normalization. `kivi_memory.enrichment.temporal` interprets date/time/recurrence/calendar meaning.

Current temporal fields:

```text
temporal_kind, valid_from_hint, valid_to_hint, event_time,
temporal_precision, recurrence, recurrence_specifics, calendar_event
```

Calendar projection is derived from semantic memory into `calendar_events` and `calendar_event_entities`. Recurring events are stored as series rows, and `CalendarTool` expands occurrences at read time. Calendar is a projection, not an independent source of truth.

## Redis Short-Term Context

`ThreadEpisode` stores one visible USER/ASSISTANT turn. Redis keys include:

```text
kivi:thread:{thread_id}:episodes
kivi:threads:episode_index
kivi:thread:{thread_id}:writeback_state
kivi:thread:{thread_id}:writeback_lock
```

Current defaults include `KIVI_THREAD_CONTEXT_EPISODES=6`, `KIVI_THREAD_CONTEXT_MAX_TOKENS=2500`, `KIVI_THREAD_ACTIVE_WINDOW_EPISODES=20`, and `KIVI_THREAD_ACTIVE_WINDOW_TOKENS=8000`. Older same-thread context can be retrieved by `redis_thread_history.search`.

## Redis To Durable Writeback

Implemented flow:

```text
ThreadEpisodes
-> select_active_thread_window
-> FlushPolicy
-> FlushCoordinator
-> MemoryEpisodeBuilder
-> MemoryWritebackWorker
-> MemoryPipeline
-> cursor update in WritebackStateStore
```

Eligible USER messages can create memory. Assistant messages and overlap messages are context only. Ingestion IDs are deterministic from thread id and stream boundaries, and cursors advance only after pipeline success.

## Tools

| Tool | Purpose | When Router Uses It |
|---|---|---|
| `semantic_memory.search` | Persistent facts/preferences/projects/relationships | Stored knowledge questions |
| `calendar.get_schedule` | Events, availability, deadlines, time ranges | Schedule/calendar questions |
| `redis_thread_history.search` | Older same-thread conversation | “Earlier in this thread” questions |
| `memory.control` | Inspect/explain/correct/remove/forget memory | Explicit memory-control requests |
| `web.search` | Current/external web information | Latest/current facts not in memory |

Recent Redis context is baseline context, not a routed tool.

## Example Queries

- `Who handles Atlas?`
- `Who leads Helios now?`
- `Do I prefer all meetings in the morning?`
- `What do I have next Tuesday?`
- `What hotel did I book for Bengaluru?` should refuse to invent if no evidence exists.

## Corpus

Development corpus: `data/kivi_dev_corpus_500.jsonl`. It is a chronological JSONL corpus with records like:

```json
{
  "record_id": "rec_0001",
  "raw_asr": "...",
  "formatted_text": "...",
  "timestamp": "2026-01-05T08:34:00+05:30",
  "metadata": {}
}
```

It covers facts, people/projects, preferences, uncertainty, attribution, corrections, supersessions, negations, recurring events, deadlines, irrelevant chatter, questions that should not become memory, aliases/coreference, and distributed information. It is development/evaluation data, not the hidden evaluator corpus.

## Evaluation

Generated artifacts:

- `evaluation/results.json`
- `evaluation/results_summary.json`
- `evaluation/corpus_ingestion_report.json`
- `evaluation/converted_memory_episodes.jsonl`

Current `results_summary.json` is from a one-case smoke run:

```text
total: 1
passed: 1
failed: 0
pass_rate: 1.0
average_latency_ms: 17845.8584
```

Current ingestion report is from a clean 3-record smoke run:

```text
processed: 3
successful: 3
failed: 0
ADD: 3
REINFORCE: 0
SUPERSEDE: 0
RETRACT: 0
average_latency_seconds: 35.0703
```

Run the full 15-case evaluation with:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\run_evaluation.py
```

## Inspectability

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\inspect_memory_state.py --query Atlas --limit 25
.\.venv\Scripts\python.exe scripts\list_memories.py --limit 25
.\.venv\Scripts\python.exe scripts\get_memory.py --memory-id <memory_id>
```

Inspection output includes memory status, evidence, source record IDs, source spans, ledger events, arguments, temporal fields, and calendar rows.

## Technology And Models

| Component | Current Implementation |
|---|---|
| Frontend | Vite + React, `frontend`, default port `5173` |
| Backend | FastAPI, `kivi_memory.api.chat:app`, default port `8000` |
| Database | PostgreSQL + pgvector |
| Cache/context | Redis |
| Local model runtime | Ollama |
| Semantic Compiler | `qwen3.5:9b`, `think=True` |
| Temporal gate | `qwen3.5:9b`, `think=False` |
| Temporal reasoning | `qwen3.5:9b` |
| Temporal normalization | `qwen3.5:9b` |
| Reconciliation | `qwen3.5:9b`, `think=False`, temperature `0` |
| Read router | `qwen3.5:4b`, `think=False` |
| Final answer | Sarvam `sarvam-105b`, `reasoning_effort=low` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Web search | Tavily by default; DuckDuckGo path exists |

## Model-Driven Vs Deterministic

Model-driven: Semantic Compiler, temporal gate/reasoning/normalization, reconciliation judge, read-side tool router, final Sarvam answer generation.

Deterministic: schema validation, ledger mutation, Redis active-window selection, writeback cursor/idempotency, context construction, calendar occurrence expansion, projection orchestration.

## Limitations

- Local Ollama inference can be slow.
- Development corpus is synthetic/test data.
- Model-assisted extraction/reconciliation can still err.
- Local PostgreSQL/Redis/Ollama assumptions.
- Web search and final answers depend on external providers when invoked.

## Repository Structure

```text
src/kivi_memory/        backend memory, read, writeback, retrieval, API code
frontend/               Vite/React UI
scripts/                import, ingest, inspect, reset, migration, evaluation helpers
migrations/             PostgreSQL projection/retrieval migrations
data/                   development corpus
evaluation/             generated ingestion/evaluation artifacts
tests/                  pytest suite
docker-compose.yml      local PostgreSQL + Redis
RUN.md                  reviewer runbook
.env.example            safe environment template
```

## AI Usage Disclosure

Kivi uses AI models in the product: Ollama models for semantic compilation, temporal interpretation, reconciliation, and routing; Sarvam for final answers; sentence-transformers for embeddings. Development assistance was used to implement and document the repository. Product Positioning and Product Vision are not generated here and must be written personally by the applicant if required.

## Security And Privacy

Secrets are loaded from environment variables. `.env` is ignored by Git and must not be committed. PostgreSQL, Redis, and Ollama are local by default. External providers receive data only when invoked: Sarvam for final answers and Tavily/DuckDuckGo for web search.
