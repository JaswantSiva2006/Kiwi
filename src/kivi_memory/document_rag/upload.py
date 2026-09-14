from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from .embedding_service import DocumentEmbeddingService
from .ingestion import ingest_parsed_document
from .repository import DocumentRagRepository
from kivi_memory.PDF_parse.parser import parse_pdf_document


UPLOAD_ROOT = Path("data/document_uploads")


@dataclass(frozen=True)
class DocumentUploadResult:
    document_id: str
    filename: str
    sha256: str
    status: str
    duplicate: bool
    chunks_indexed: int
    parse_status: str | None
    embedding_time: float
    db_write_time: float
    total_time: float
    error: str | None = None


def ingest_uploaded_pdf(
    *,
    filename: str,
    stream: BinaryIO,
    repository: DocumentRagRepository | None = None,
    embedding_service: DocumentEmbeddingService | None = None,
    upload_root: Path = UPLOAD_ROOT,
) -> DocumentUploadResult:
    started = time.perf_counter()
    safe_name = _safe_pdf_filename(filename)
    upload_root.mkdir(parents=True, exist_ok=True)
    saved_path = upload_root / f"{uuid4().hex}_{safe_name}"
    sha256 = _save_and_hash(stream, saved_path)
    repository = repository or DocumentRagRepository()

    existing = repository.find_document_by_sha256(sha256)
    if existing is not None and existing.get("status") == "READY":
        return DocumentUploadResult(
            document_id=str(existing["document_id"]),
            filename=str(existing.get("filename") or safe_name),
            sha256=sha256,
            status="READY",
            duplicate=True,
            chunks_indexed=0,
            parse_status=None,
            embedding_time=0.0,
            db_write_time=0.0,
            total_time=time.perf_counter() - started,
        )

    parsed = parse_pdf_document(saved_path)
    parse_status = parsed["document"]["parse_status"]
    if parse_status != "OK":
        return DocumentUploadResult(
            document_id=parsed["document"]["document_id"],
            filename=safe_name,
            sha256=sha256,
            status="FAILED",
            duplicate=False,
            chunks_indexed=0,
            parse_status=parse_status,
            embedding_time=0.0,
            db_write_time=0.0,
            total_time=time.perf_counter() - started,
            error="PDF requires OCR or could not be parsed.",
        )

    parsed["document"]["filename"] = safe_name
    result = ingest_parsed_document(parsed, repository=repository, embedding_service=embedding_service)
    return DocumentUploadResult(
        document_id=result.document_id,
        filename=safe_name,
        sha256=sha256,
        status="READY",
        duplicate=result.duplicate,
        chunks_indexed=result.chunks_indexed,
        parse_status=parse_status,
        embedding_time=result.embedding_time,
        db_write_time=result.db_write_time,
        total_time=time.perf_counter() - started,
    )


def _save_and_hash(stream: BinaryIO, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)
    return digest.hexdigest()


def _safe_pdf_filename(filename: str) -> str:
    name = Path(filename or "document.pdf").name
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF uploads are supported.")
    return name
