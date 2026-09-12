# Kivi Reviewer Runbook

Review mode: **LOCAL**.

Run commands from the repository root in PowerShell.

## Prerequisites

- Python with virtualenv support
- Node/npm
- Docker Desktop or local PostgreSQL + Redis
- Ollama
- Sarvam API key for final answers
- Tavily API key only if testing web search

## Setup

```powershell
git clone <repo-url>
cd <repo>

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

cd frontend
npm install
cd ..

Copy-Item .env.example .env
```

Edit `.env` and set at least `SARVAM_API_KEY`. Do not commit `.env`.

## Infrastructure

Start PostgreSQL and Redis:

```powershell
docker compose up -d postgres redis
```

Apply migrations:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\apply_hnsw_migration.py
.\.venv\Scripts\python.exe scripts\apply_graph_migration.py
.\.venv\Scripts\python.exe scripts\apply_read_retrieval_migration.py
.\.venv\Scripts\python.exe scripts\apply_calendar_migration.py
.\.venv\Scripts\python.exe scripts\apply_calendar_recurrence_migration.py
```

Pull/check Ollama models:

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama list
.\.venv\Scripts\python.exe scripts\check_ollama.py
```

## Reset

Destructive: clears Kivi semantic-memory tables and generated reports.

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\reset_memory_state.py --yes
```

## Import Development Corpus

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\import_corpus.py `
  --input data\kivi_dev_corpus_500.jsonl `
  --output evaluation\converted_memory_episodes.jsonl
```

## Ingest Development Corpus

Smoke:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\ingest_corpus.py `
  --input data\kivi_dev_corpus_500.jsonl `
  --limit 3 `
  --continue-on-error
```

Full:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\ingest_corpus.py `
  --input data\kivi_dev_corpus_500.jsonl `
  --continue-on-error
```

Resume from chronological index:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_corpus.py `
  --input data\kivi_dev_corpus_500.jsonl `
  --start-from 126 `
  --continue-on-error
```

Reports:

- `evaluation/converted_memory_episodes.jsonl`
- `evaluation/corpus_ingestion_report.json`

## Hidden Evaluator Corpus

Accepted JSONL shape:

```json
{"record_id":"rec_0001","raw_asr":"...","formatted_text":"...","timestamp":"2026-01-05T08:34:00+05:30","metadata":{}}
```

Commands:

```powershell
$env:PYTHONPATH='src'

.\.venv\Scripts\python.exe scripts\import_corpus.py `
  --input <path-to-hidden-corpus.jsonl> `
  --output evaluation\hidden_memory_episodes.jsonl

.\.venv\Scripts\python.exe scripts\ingest_corpus.py `
  --input <path-to-hidden-corpus.jsonl> `
  --continue-on-error
```

Options supported by ingestion:

- `--limit N`
- `--start-from N`
- `--continue-on-error`
- `--debug-tracebacks`

## Inspect Learned Memory

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\inspect_memory_state.py --limit 25
.\.venv\Scripts\python.exe scripts\inspect_memory_state.py --query Atlas --limit 25
.\.venv\Scripts\python.exe scripts\list_memories.py --status ACTIVE --limit 25
.\.venv\Scripts\python.exe scripts\get_memory.py --memory-id <memory_id>
```

These show active/superseded/retracted memories, source evidence, source record IDs, ledger events, temporal fields, and calendar rows.

## Start Backend

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m uvicorn kivi_memory.api.chat:app --host 127.0.0.1 --port 8000
```

Backend URL: `http://127.0.0.1:8000`

## Start Frontend

```powershell
cd frontend
npm run dev
```

Frontend URL: Vite default `http://127.0.0.1:5173`

## Manual Queries

Try after ingesting the development corpus:

```text
Who leads Helios now, and what is Arjun's role?
Do I prefer all meetings in the morning?
Did I already buy the Pixel, or did I only decide I prefer it?
What is my normal Payments review schedule, and was there any exception?
What hotel did I book for Bengaluru?
```

For the hotel question, expected behavior is to refuse to invent if no booking evidence exists.

## Evaluation

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

Smoke:

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py --limit 1
```

Outputs:

- `evaluation/results.json`
- `evaluation/results_summary.json`

## Reset And Repeat Hidden-Corpus Testing

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\reset_memory_state.py --yes
.\.venv\Scripts\python.exe scripts\import_corpus.py --input <path-to-hidden-corpus.jsonl> --output evaluation\hidden_memory_episodes.jsonl
.\.venv\Scripts\python.exe scripts\ingest_corpus.py --input <path-to-hidden-corpus.jsonl> --continue-on-error
.\.venv\Scripts\python.exe scripts\inspect_memory_state.py --limit 25
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

## Troubleshooting

- Wrong Python interpreter: use `.\.venv\Scripts\python.exe`, not system Python.
- Postgres check:
  ```powershell
  docker compose ps postgres
  ```
- Redis check:
  ```powershell
  docker compose ps redis
  ```
- Ollama missing model:
  ```powershell
  ollama list
  ollama pull qwen3.5:9b
  ollama pull qwen3.5:4b
  ```
- `sentence_transformers` missing usually means the wrong Python environment or incomplete `pip install -r requirements.txt`.
- PowerShell multiline commands use backtick `` ` ``, not `\`.
- Web search needs `TAVILY_API_KEY` when `KIVI_WEB_PROVIDER=tavily`.
