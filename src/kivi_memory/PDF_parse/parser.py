from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .chunker import chunk_sections
from .heading_detector import detect_headings
from .schemas import TextBlock, output_document
from .section_builder import build_sections
from .validator import validate_output


def parse_pdf_document(input_path: str | Path, *, include_heading_diagnostics: bool = False) -> dict[str, Any]:
    path = Path(input_path)
    document_id = _document_id(path)
    sha256 = _sha256(path)
    warnings: list[str] = []
    try:
        pdf = fitz.open(path)
    except Exception as exc:
        return _error_output(path, document_id, sha256, f"open_error:{exc}")

    try:
        blocks = _extract_blocks(pdf)
        page_count = pdf.page_count
    except Exception as exc:
        pdf.close()
        return _error_output(path, document_id, sha256, f"parse_error:{exc}")
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    characters = sum(len(block.text) for block in blocks)
    if _requires_ocr(page_count, blocks, characters):
        warnings.append("ocr_required_low_text_density")
        return {
            "document": output_document(
                document_id=document_id,
                filename=path.name,
                sha256=sha256,
                page_count=page_count,
                parse_status="OCR_REQUIRED",
            ),
            "chunks": [],
            "stats": {
                "characters_extracted": characters,
                "blocks_extracted": len(blocks),
                "chunks_created": 0,
                "average_tokens_per_chunk": 0,
            },
            "warnings": warnings,
        }

    diagnostics = detect_headings(blocks)
    sectioned = build_sections(blocks)
    chunks = chunk_sections(sectioned, document_id=document_id, filename=path.name)
    output = {
        "document": output_document(
            document_id=document_id,
            filename=path.name,
            sha256=sha256,
            page_count=page_count,
            parse_status="OK",
        ),
        "chunks": chunks,
        "stats": {
            "characters_extracted": characters,
            "blocks_extracted": len(blocks),
            "chunks_created": len(chunks),
            "average_tokens_per_chunk": _average_tokens(chunks),
        },
        "warnings": warnings,
    }
    output["warnings"] = warnings + validate_output(output, {block.block_id for block in blocks})
    if include_heading_diagnostics:
        output["heading_diagnostics"] = diagnostics
    return output


def write_json(output: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_blocks(pdf: fitz.Document) -> list[TextBlock]:
    extracted: list[TextBlock] = []
    order = 0
    for page_index in range(pdf.page_count):
        page = pdf.load_page(page_index)
        raw = page.get_text("dict", sort=True)
        previous_y: float | None = None
        page_blocks: list[TextBlock] = []
        for raw_block in raw.get("blocks", []):
            if raw_block.get("type") != 0:
                continue
            spans = [
                span
                for line in raw_block.get("lines", [])
                for span in line.get("spans", [])
                if _cleanup_text(span.get("text", ""))
            ]
            text = _cleanup_text(" ".join(span.get("text", "") for span in spans))
            if not text:
                continue
            sizes = [float(span.get("size", 0)) for span in spans if span.get("size")]
            fonts = sorted({span.get("font", "") for span in spans if span.get("font")})
            flags = 0
            for span in spans:
                flags |= int(span.get("flags", 0))
            bbox = [round(float(value), 2) for value in raw_block.get("bbox", [0, 0, 0, 0])]
            block = TextBlock(
                block_id=f"p{page_index + 1}:b{len(page_blocks):04d}",
                text=text,
                page_number=page_index + 1,
                bbox=bbox,
                font_size=round(max(sizes) if sizes else 0.0, 2),
                fonts=fonts,
                flags=flags,
                order=order,
            )
            if previous_y is not None:
                block.gap_before = max(0.0, block.bbox[1] - previous_y)
            previous_y = block.bbox[3]
            page_blocks.append(block)
            order += 1
        extracted.extend(sorted(page_blocks, key=lambda b: (round(b.bbox[1], 1), round(b.bbox[0], 1), b.order)))
    return extracted


def _requires_ocr(page_count: int, blocks: list[TextBlock], characters: int) -> bool:
    if page_count == 0:
        return True
    if not blocks:
        return True
    return characters / page_count < 80 or len(blocks) / page_count < 2


def _cleanup_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _average_tokens(chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0
    return round(sum(chunk["token_count"] for chunk in chunks) / len(chunks))


def _error_output(path: Path, document_id: str, sha256: str, warning: str) -> dict[str, Any]:
    return {
        "document": output_document(
            document_id=document_id,
            filename=path.name,
            sha256=sha256,
            page_count=0,
            parse_status="ERROR",
        ),
        "chunks": [],
        "stats": {
            "characters_extracted": 0,
            "blocks_extracted": 0,
            "chunks_created": 0,
            "average_tokens_per_chunk": 0,
        },
        "warnings": [warning],
    }
