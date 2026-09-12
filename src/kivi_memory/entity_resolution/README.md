# Kivi Entity Resolution

This package contains the first write-side entity-resolution block. It is deliberately deterministic and does not create entities during resolution.

Flow:

```text
incoming entity mention
-> normalize
-> exact/fuzzy candidate lookup
-> collapse aliases by entity
-> rank candidates
-> MATCHED / AMBIGUOUS / NEW
```

Responsibilities:

- `config.py`: shared defaults for the database URL and scoring thresholds.
- `normalizer.py`: conservative name normalization used for mentions, canonical names, and aliases.
- `repository.py`: PostgreSQL access for exact/fuzzy lookup and low-level entity/alias writes.
- `candidates.py`: candidate generation, alias collapse by `entity_id`, type-match annotation, and ranking.
- `resolver.py`: final read-only decision layer returning `MATCHED`, `AMBIGUOUS`, or `NEW`.
- `orchestrator.py`: write-aware helper that creates an entity only when the decision is `NEW`.
- `writes.py`: small public wrappers for `create_entity` and `add_entity_alias`.

Manual scripts:

- `scripts/test_entity_candidates.py`: prints ranked lookup candidates.
- `scripts/test_entity_resolution.py`: prints the final resolution decision plus candidates.
- `scripts/add_entity.py`: creates an entity and canonical alias.
- `scripts/add_entity_alias.py`: idempotently adds an alias to an existing entity.
