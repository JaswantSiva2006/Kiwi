from __future__ import annotations

import pytest

from kivi_memory.entity_resolution import EntityNotFoundError
from kivi_memory.entity_resolution.repository import EntityRepository
from kivi_memory.entity_resolution.writes import add_entity_alias, create_entity


class FakeDatabase:
    def __init__(self) -> None:
        self.entities: dict[str, dict] = {}
        self.aliases: dict[str, dict] = {}
        self.next_entity = 1
        self.next_alias = 1
        self.fail_alias_insert = False

    def connect(self, database_url: str):
        return FakeConnection(self)

    def snapshot(self):
        return (
            {key: value.copy() for key, value in self.entities.items()},
            {key: value.copy() for key, value in self.aliases.items()},
            self.next_entity,
            self.next_alias,
        )

    def restore(self, snapshot) -> None:
        self.entities, self.aliases, self.next_entity, self.next_alias = snapshot


class FakeConnection:
    def __init__(self, db: FakeDatabase) -> None:
        self.db = db
        self.rolled_back = False
        self.committed = False
        self._snapshot = None

    def __enter__(self):
        self._snapshot = self.db.snapshot()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.committed = True
        else:
            self.rolled_back = True
            self.db.restore(self._snapshot)

    def cursor(self):
        return FakeCursor(self.db)


class FakeCursor:
    def __init__(self, db: FakeDatabase) -> None:
        self.db = db
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, sql: str, params: tuple) -> None:
        normalized_sql = " ".join(sql.split()).lower()
        if normalized_sql.startswith("insert into entities"):
            entity_id = f"entity-{self.db.next_entity}"
            self.db.next_entity += 1
            row = {
                "entity_id": entity_id,
                "canonical_name": params[0],
                "normalized_name": params[1],
                "entity_type": params[2],
            }
            self.db.entities[entity_id] = row
            self.result = row
            return

        if normalized_sql.startswith("select 1 from entities"):
            self.result = {"exists": 1} if params[0] in self.db.entities else None
            return

        if normalized_sql.startswith("insert into entity_aliases"):
            if self.db.fail_alias_insert:
                raise RuntimeError("alias insert failed")

            key = (params[0], params[2])
            for row in self.db.aliases.values():
                if (row["entity_id"], row["normalized_alias"]) == key:
                    self.result = row
                    return

            alias_id = f"alias-{self.db.next_alias}"
            self.db.next_alias += 1
            row = {
                "alias_id": alias_id,
                "entity_id": params[0],
                "alias": params[1],
                "normalized_alias": params[2],
                "confidence": params[3],
                "alias_source": params[4],
                "source_episode_id": params[5],
            }
            self.db.aliases[alias_id] = row
            self.result = row
            return

        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self.result


@pytest.fixture
def fake_db(monkeypatch) -> FakeDatabase:
    db = FakeDatabase()
    monkeypatch.setattr("kivi_memory.entity_resolution.repository.connect", db.connect)
    return db


def test_create_entity_also_creates_canonical_alias(fake_db: FakeDatabase) -> None:
    entity = EntityRepository().create_entity("Priya Nair", "PERSON", "ep-1")

    assert entity.entity_id == "entity-1"
    assert entity.canonical_name == "Priya Nair"
    assert entity.normalized_name == "priya nair"
    assert fake_db.aliases["alias-1"] == {
        "alias_id": "alias-1",
        "entity_id": "entity-1",
        "alias": "Priya Nair",
        "normalized_alias": "priya nair",
        "confidence": 1.0,
        "alias_source": "CANONICAL",
        "source_episode_id": "ep-1",
    }


def test_entity_and_canonical_alias_are_atomic(fake_db: FakeDatabase) -> None:
    fake_db.fail_alias_insert = True

    with pytest.raises(RuntimeError, match="alias insert failed"):
        EntityRepository().create_entity("Priya Nair", "PERSON")

    assert fake_db.entities == {}
    assert fake_db.aliases == {}


def test_create_entity_reuses_normalization(fake_db: FakeDatabase) -> None:
    entity = EntityRepository().create_entity("  Priya,\tNair!!  ", "PERSON")

    assert entity.normalized_name == "priya nair"
    assert fake_db.aliases["alias-1"]["normalized_alias"] == "priya nair"


def test_add_new_alias_works(fake_db: FakeDatabase) -> None:
    entity = EntityRepository().create_entity("Priya Nair", "PERSON")

    alias = EntityRepository().add_entity_alias(entity.entity_id, "Priya", 0.8, "OBSERVED", "ep-2")

    assert alias.alias == "Priya"
    assert alias.normalized_alias == "priya"
    assert alias.confidence == 0.8
    assert alias.alias_source == "OBSERVED"
    assert alias.source_episode_id == "ep-2"


def test_same_normalized_alias_for_same_entity_is_idempotent(fake_db: FakeDatabase) -> None:
    entity = EntityRepository().create_entity("Priya Nair", "PERSON")

    first = EntityRepository().add_entity_alias(entity.entity_id, "Priya", 0.8, "OBSERVED")
    second = EntityRepository().add_entity_alias(entity.entity_id, " priya!! ", 0.3, "OTHER")

    assert first.alias_id == second.alias_id
    assert len(fake_db.aliases) == 2
    assert second.alias == "Priya"
    assert second.confidence == 0.8
    assert second.alias_source == "OBSERVED"


def test_different_entities_can_share_same_normalized_alias(fake_db: FakeDatabase) -> None:
    first = EntityRepository().create_entity("Priya Nair", "PERSON")
    second = EntityRepository().create_entity("Priya Sharma", "PERSON")

    first_alias = EntityRepository().add_entity_alias(first.entity_id, "Priya")
    second_alias = EntityRepository().add_entity_alias(second.entity_id, "Priya")

    assert first_alias.alias_id != second_alias.alias_id
    assert first_alias.normalized_alias == second_alias.normalized_alias == "priya"


def test_add_alias_to_nonexistent_entity_fails_cleanly(fake_db: FakeDatabase) -> None:
    with pytest.raises(EntityNotFoundError, match="entity does not exist"):
        EntityRepository().add_entity_alias("missing", "Priya")


@pytest.mark.parametrize(
    ("name", "entity_type"),
    [("", "PERSON"), ("   ", "PERSON"), ("Priya", ""), ("Priya", "   ")],
)
def test_create_entity_rejects_empty_names_and_types(
    fake_db: FakeDatabase,
    name: str,
    entity_type: str,
) -> None:
    with pytest.raises(ValueError):
        create_entity(name, entity_type, repository=EntityRepository())


@pytest.mark.parametrize("alias", ["", "   ", "!!!"])
def test_add_entity_alias_rejects_empty_aliases(fake_db: FakeDatabase, alias: str) -> None:
    entity = EntityRepository().create_entity("Priya Nair", "PERSON")

    with pytest.raises(ValueError):
        add_entity_alias(entity.entity_id, alias, repository=EntityRepository())
