from __future__ import annotations

import json

from kivi_memory.common.schemas import MemoryEpisode
from kivi_memory.corpus_import import convert_jsonl_file, read_corpus_jsonl


def test_three_record_jsonl_converts_target_only_episodes_with_provenance(tmp_path) -> None:
    input_path = tmp_path / "dev_corpus.jsonl"
    output_path = tmp_path / "dev_memory_episodes.jsonl"
    records = [
        {
            "record_id": "rec_001",
            "raw_asr": "i met priya about atlas",
            "formatted_text": "I met Priya about Atlas.",
            "timestamp": "2026-08-01T10:00:00+05:30",
            "metadata": {"timezone": "Asia/Kolkata", "locale": "en-IN", "source": "dictation"},
        },
        {
            "record_id": "rec_002",
            "raw_asr": "she said she'll handle the review",
            "formatted_text": "She said she'll handle the review.",
            "timestamp": "2026-08-01T10:01:00+05:30",
            "metadata": {"timezone": "Asia/Kolkata", "locale": "en-IN", "source": "dictation"},
        },
        {
            "record_id": "rec_003",
            "raw_asr": "the review is next friday",
            "formatted_text": "The review is next Friday.",
            "timestamp": "2026-08-01T10:02:00+05:30",
            "metadata": {"timezone": "Asia/Kolkata", "locale": "en-IN", "source": "dictation"},
        },
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    parsed, errors = read_corpus_jsonl(input_path)
    result = convert_jsonl_file(input_path, output_path)

    assert len(parsed) == 3
    assert errors == []
    assert result.errors == []
    assert len(result.episodes) == 3

    lines = output_path.read_text(encoding="utf-8").splitlines()
    middle = MemoryEpisode.model_validate(json.loads(lines[1]))

    assert middle.episode_id == "corpus-rec-002"
    assert len(middle.messages) == 1
    assert middle.messages[0].text == "She said she'll handle the review."
    assert middle.messages[0].context_only is False
    assert middle.messages[0].memory_eligible is True
    assert middle.messages[0].source_thread_episode_id == "rec_002"

    provenance = json.loads(middle.tool_context[0].content)
    assert provenance["target_record_id"] == "rec_002"
    assert provenance["raw_asr"] == "she said she'll handle the review"
    assert provenance["metadata"]["source"] == "dictation"
    assert "formatted_text" not in provenance


def test_required_fields_fail_cleanly(tmp_path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(json.dumps({"record_id": "rec_001", "raw_asr": "missing fields"}) + "\n", encoding="utf-8")

    _records, errors = read_corpus_jsonl(input_path)

    assert errors
    assert errors[0]["line"] == 1
