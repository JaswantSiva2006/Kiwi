from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ParseStatus = Literal["OK", "OCR_REQUIRED", "ERROR"]
BlockKind = Literal["HEADING", "BODY", "CAPTION", "LIST", "OTHER"]
ChunkType = Literal["TEXT", "TABLE", "CAPTION", "OTHER"]


@dataclass(frozen=True)
class TextSpan:
    text: str
    font: str
    size: float
    flags: int


@dataclass
class TextBlock:
    block_id: str
    text: str
    page_number: int
    bbox: list[float]
    font_size: float
    fonts: list[str]
    flags: int
    order: int
    gap_before: float = 0.0
    kind: BlockKind = "BODY"
    heading_score: int = 0
    heading_signals: list[str] = field(default_factory=list)


@dataclass
class SectionedBlock:
    block: TextBlock
    section_title: str | None
    section_path: list[str]


def output_document(
    *,
    document_id: str,
    filename: str,
    sha256: str,
    page_count: int,
    parse_status: ParseStatus,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "filename": filename,
        "sha256": sha256,
        "page_count": page_count,
        "parser": "pymupdf",
        "parse_status": parse_status,
    }
