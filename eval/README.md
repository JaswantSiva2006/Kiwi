# Kivi Semantic Compiler Eval

This directory contains small adversarial episodes for manually inspecting the local semantic compiler baseline.

Run:

```bash
python eval/run_eval.py
```

The runner calls the local Ollama-backed compiler for each `eval/episodes/*.json`, writes actual outputs to `eval/results/`, and checks only lightweight deterministic expectations from `eval/expected/`. It does not use an LLM judge and does not require exact canonical wording.
