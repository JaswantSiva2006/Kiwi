from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from kivi_memory.document_rag import upload as upload_module
from kivi_memory.document_rag.upload import ingest_uploaded_pdf


def test_upload_duplicate_ready_skips_parse_and_embedding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = FakeRepository()

    def fail_parse(path: Path) -> dict[str, Any]:
        raise AssertionError("duplicate READY document should not be parsed")

    monkeypatch.setattr(upload_module, "parse_pdf_document", fail_parse)

    result = ingest_uploaded_pdf(
        filename="paper.pdf",
        stream=BytesIO(b"%PDF-duplicate"),
        repository=repository,
        upload_root=tmp_path,
    )

    assert result.status == "READY"
    assert result.duplicate is True
    assert result.document_id == "doc-existing"
    assert result.chunks_indexed == 0


class FakeRepository:
    def find_document_by_sha256(self, sha256: str) -> dict[str, Any]:
        return {
            "document_id": "doc-existing",
            "filename": "paper.pdf",
            "sha256": sha256,
            "page_count": 3,
            "status": "READY",
        }
