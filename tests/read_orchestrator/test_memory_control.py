from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from kivi_memory.long_term_retrieval import RetrievalDiagnostics, RetrievalResult
from kivi_memory.long_term_retrieval.models import HydratedRetrievedMemory
from kivi_memory.read_orchestrator.memory_control import MemoryControlStatus, MemoryControlTool


class FakeRetriever:
    def __init__(self, memories) -> None:
        self.memories = memories
        self.calls = []

    def __call__(self, *, query_text, top_k=12):
        self.calls.append({"query_text": query_text, "top_k": top_k})
        return RetrievalResult(
            query=query_text,
            resolved_query_entities=[],
            memories=self.memories,
            diagnostics=RetrievalDiagnostics(),
        )


class FakeRepo:
    def __init__(self) -> None:
        self.locked = []
        self.status_updates = []
        self.events = []

    def connect(self):
        return FakeConn()

    def lock_active_memory(self, cur, memory_id):
        self.locked.append(memory_id)

    def update_memory_status_in_transaction(self, cur, *, memory_id, status):
        self.status_updates.append((memory_id, status))

    def insert_event_in_transaction(self, cur, *, memory_id, event_type, payload):
        self.events.append((memory_id, event_type, payload))


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self


class FakePipeline:
    def __init__(self) -> None:
        self.episodes = []

    def process(self, episode):
        self.episodes.append(episode)
        return type(
            "Result",
            (),
            {
                "output": {
                    "assertions": [
                        {
                            "mutation_result": {
                                "target_memory_ids": ["old-1"],
                                "created_memory_id": "new-1",
                            }
                        }
                    ]
                }
            },
        )()


def test_inspect_returns_memories_and_provenance() -> None:
    tool = MemoryControlTool(retriever=FakeRetriever([memory("m1", "Priya handles Atlas.")]), memory_pipeline=FakePipeline())

    result = tool.execute(query="What do you remember about Priya?", thread_id="t1", current_datetime=now(), timezone="Asia/Kolkata")

    assert result.status == MemoryControlStatus.OK
    assert result.memories[0]["canonical_text"] == "Priya handles Atlas."
    assert result.provenance[0]["evidence"][0]["source_text"] == "Priya handles Atlas."


def test_ambiguous_forget_does_not_mutate() -> None:
    repo = FakeRepo()
    tool = MemoryControlTool(
        retriever=FakeRetriever([memory("m1", "User prefers morning meetings."), memory("m2", "User prefers short meetings.")]),
        repository=repo,
        memory_pipeline=FakePipeline(),
    )

    result = tool.execute(query="Forget my meeting preference.", thread_id="t1", current_datetime=now(), timezone="Asia/Kolkata")

    assert result.status == MemoryControlStatus.NEEDS_CLARIFICATION
    assert repo.status_updates == []


def test_broad_forget_requires_confirmation() -> None:
    repo = FakeRepo()
    tool = MemoryControlTool(retriever=FakeRetriever([memory("m1", "Priya handles Atlas.")]), repository=repo, memory_pipeline=FakePipeline())

    result = tool.execute(query="Forget everything about Priya.", thread_id="t1", current_datetime=now(), timezone="Asia/Kolkata")

    assert result.status == MemoryControlStatus.NEEDS_CONFIRMATION
    assert repo.status_updates == []


def test_forget_marks_memory_retracted_and_records_event(monkeypatch) -> None:
    repo = FakeRepo()
    monkeypatch.setattr("kivi_memory.read_orchestrator.memory_control.cancel_calendar_event_for_memory", lambda cur, *, memory_id: 0)
    tool = MemoryControlTool(retriever=FakeRetriever([memory("m1", "The user prefers morning meetings.")]), repository=repo, memory_pipeline=FakePipeline())

    result = tool.execute(query="Forget that I prefer morning meetings.", thread_id="t1", current_datetime=now(), timezone="Asia/Kolkata")

    assert result.status == MemoryControlStatus.APPLIED
    assert repo.locked == ["m1"]
    assert repo.status_updates == [("m1", "RETRACTED")]
    assert repo.events[0][1] == "MEMORY_RETRACTED"


def test_correct_uses_memory_pipeline() -> None:
    pipeline = FakePipeline()
    tool = MemoryControlTool(retriever=FakeRetriever([memory("old-1", "Rohit handles Atlas.")]), memory_pipeline=pipeline)

    result = tool.execute(
        query="No, Rohit doesn't handle Atlas anymore. Priya does.",
        thread_id="t1",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    )

    assert result.status == MemoryControlStatus.APPLIED
    assert result.affected_memory_ids == ["new-1", "old-1"]
    assert pipeline.episodes[0].messages[0].memory_eligible is True


def memory(memory_id: str, text: str) -> HydratedRetrievedMemory:
    return HydratedRetrievedMemory(
        memory_id=memory_id,
        canonical_text=text,
        status="ACTIVE",
        version=1,
        rrf_score=0.1,
        retrieval_sources=["LEXICAL"],
        branch_ranks={"LEXICAL": 1},
        branch_scores={"LEXICAL": 0.5},
        subject={"text": "Priya", "entity_id": None, "entity_type": "PERSON"},
        predicate_type="TEST",
        memory_type="TEST",
        modality=None,
        polarity=None,
        certainty=None,
        explicitness=None,
        attributed_to="user",
        temporal={},
        arguments=[],
        evidence=[{"episode_id": "ep1", "message_id": "u1", "source_text": text, "observed_at": "2026-09-11T10:00:00+05:30"}],
    )


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
