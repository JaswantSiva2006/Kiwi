# Submission Audit

Generated at final documentation hardening pass.

| Deliverable | Status | Path / Note |
|---|---|---|
| Product Positioning document | MISSING | Applicant must personally write this before submission. |
| Product Vision document | MISSING | Applicant must personally write this before submission. |
| complete source code | PASS | `src/`, `frontend/`, `scripts/`, `tests/` |
| working frontend | PASS | `frontend/package.json`, Vite/React app |
| working backend | PASS | `src/kivi_memory/api/chat.py`, FastAPI app |
| DB schema/migrations | PASS | `migrations/`, `scripts/apply_*_migration.py` |
| reproducible seed/development data | PASS | `data/kivi_dev_corpus_500.jsonl`, `data/dev_corpus_tiny.jsonl` |
| ~500-record corpus | PASS | `data/kivi_dev_corpus_500.jsonl` |
| hidden-corpus importer | PASS | `scripts/import_corpus.py`, `kivi_memory.corpus_import.convert_jsonl_file` |
| corpus ingestion pipeline | PASS | `scripts/ingest_corpus.py`, real `MemoryPipeline` |
| memory inspection capability | PASS | `scripts/inspect_memory_state.py`, `scripts/list_memories.py`, `scripts/get_memory.py` |
| evaluation code | PASS | `scripts/run_evaluation.py` |
| generated evaluation results | PASS | `evaluation/results.json`, `evaluation/results_summary.json` from smoke run |
| provenance/evidence artifacts | PASS | `evaluation/corpus_ingestion_report.json`, memory evidence tables exposed by inspection |
| README.md | PASS | `README.md` |
| RUN.md | PASS | `RUN.md` |
| .env.example | PASS | `.env.example` safe placeholders only |
| reset/reproducibility procedure | PASS | `scripts/reset_memory_state.py --yes`, documented in `RUN.md` |

## Smoke Verification

- Python script compile: PASS
- PostgreSQL connectivity: PASS
- Redis connectivity: PASS
- Required Ollama models available: PASS (`qwen3.5:9b`, `qwen3.5:4b`)
- Backend import: PASS
- Frontend build: PASS
- Focused pytest checks: PASS
- Clean 3-record reset/import/ingest/inspect smoke: PASS
- Real Hey Kivi read query after 3-record ingest: PASS
- Evaluation smoke command `scripts/run_evaluation.py --limit 1`: PASS

## Security Notes

- `.env` exists locally and is ignored by Git.
- `.env.example` contains placeholders only.
- `.venv`, `node_modules`, caches, and generated debug outputs are ignored.
- No Product Positioning or Product Vision content was generated.
