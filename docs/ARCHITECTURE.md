# Kivi Architecture

## Current Interactive Runtime

```text
User
  -> TurnController
  -> Redis: append USER
  -> Orchestrator
  -> Redis: read bounded current-thread context
  -> qwen3.5:4b long-term-memory checker/query generator
     -> NO
        -> Context Assembler
     -> YES
        -> Memory Retrieval
        -> VECTOR / LEXICAL / STRUCTURED / GRAPH
        -> RRF Top-K
        -> Context Assembler
  -> Sarvam API sarvam-105b MAIN HEY KIVI LLM
  -> assistant response
  -> TurnController
  -> Redis: append ASSISTANT
  -> User
```

Normal user-facing interaction enters through:

```python
TurnController.handle_user_turn(...)
```

The turn controller owns conversation writes. It appends the incoming user
message before routing/retrieval and appends the assistant response only after
successful final generation.

## Model Roles

- Router/query generator: `qwen3.5:4b`, `think=False`, `temperature=0`
- Main Hey Kivi response model: `sarvam-105b`, `reasoning_effort=low`

The main response model reads `SARVAM_API_KEY` from the environment. The key is
never logged or written to Redis.

The response model receives deterministic context only:

```text
CURRENT THREAD
----------------
<bounded prior Redis messages>

LONG-TERM MEMORY
----------------
<already-ranked retrieved memories, omitted when not needed>

CURRENT USER REQUEST
----------------
<current query>
```

No extra model-based memory selection, reranking, graph reasoning, or retrieval
is run between long-term retrieval and the main response model.

## Redis Ownership

Redis conversation writers:

- `TurnController`
- future user-visible tool runtime

Redis conversation readers:

- `TurnController`
- `llm_redis_orchestrator` memory-preparation layer
- future MemoryEpisode assembler

These components must not write Redis conversation messages:

- long-term memory retrieval
- backend memory ingestion
- semantic compiler
- reconciliation
- entity resolver

## Not Yet Implemented

```text
Redis thread messages
  -> MemoryEpisode assembler
  -> backend_memory_ingestion
  -> Postgres long-term memory update
```

The live interaction path does not synchronously invoke backend memory ingestion
or update durable semantic memory.
