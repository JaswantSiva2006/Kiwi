from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pymupdf as fitz

from kivi_memory.PDF_parse.parser import parse_pdf_document


def test_normal_paper_sections_and_schema(tmp_path: Path) -> None:
    pdf = tmp_path / "normal.pdf"
    _make_normal_pdf(pdf)

    output = parse_pdf_document(pdf)

    assert output["document"]["parse_status"] == "OK"
    assert output["chunks"]
    assert output["warnings"] == []
    assert any("1 Introduction" in chunk["section_path"] for chunk in output["chunks"])
    assert all(chunk["metadata"]["mime_type"] == "application/pdf" for chunk in output["chunks"])


def test_multi_column_reading_and_hierarchy(tmp_path: Path) -> None:
    pdf = tmp_path / "multi_column.pdf"
    _make_multi_column_pdf(pdf)

    output = parse_pdf_document(pdf, include_heading_diagnostics=True)

    assert output["document"]["parse_status"] == "OK"
    paths = [chunk["section_path"] for chunk in output["chunks"]]
    assert any(path[-1:] == ["2 Related Work"] for path in paths)
    assert "heading_diagnostics" in output


def test_report_captions_tables_cli_and_validation(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    out = tmp_path / "out.json"
    _make_report_pdf(pdf)

    result = subprocess.run(
        [sys.executable, "scripts/parse_document.py", str(pdf), "--output", str(out)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["document"]["parse_status"] == "OK"
    assert data["stats"]["chunks_created"] > 0
    assert all(chunk["page_start"] <= chunk["page_end"] for chunk in data["chunks"])


def test_low_text_pdf_returns_ocr_required(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(pdf)
    doc.close()

    output = parse_pdf_document(pdf)

    assert output["document"]["parse_status"] == "OCR_REQUIRED"
    assert output["chunks"] == []


def _make_normal_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    y = _write(page, "1 Introduction", 72, y, 16, "helv", bold=True)
    for _ in range(5):
        y = _write(page, _para("This paper introduces a practical parser for document structure."), 72, y, 11)
    y += 18
    y = _write(page, "1.1 Contributions", 72, y, 14, "helv", bold=True)
    for _ in range(4):
        y = _write(page, _para("The system keeps sections intact while producing useful chunks."), 72, y, 11)
    doc.save(path)
    doc.close()


def _make_multi_column_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    _write(page, "2 Related Work", 72, 72, 16, "helv", bold=True)
    y_left = 110
    y_right = 110
    for _ in range(7):
        y_left = _write(page, _para("Prior systems often chunk documents without respecting sections."), 72, y_left, 10)
        y_right = _write(page, _para("Column layouts require sorting by vertical and horizontal position."), 320, y_right, 10)
    doc.save(path)
    doc.close()


def _make_report_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = _write(page, "Appendix A Results", 72, 72, 17, "helv", bold=True)
    y = _write(page, "Table 1: Retrieval quality by document type.", 72, y + 10, 10)
    for _ in range(6):
        y = _write(page, _para("The report summarizes measured behavior across inputs with visible headings and tables."), 72, y, 11)
    doc.save(path)
    doc.close()


def _write(page: fitz.Page, text: str, x: int, y: float, size: int, font: str = "helv", bold: bool = False) -> float:
    fontname = "helv" if not bold else "hebo"
    max_chars = 60 if x < 300 else 34
    lines = _wrap(text, max_chars)
    line_height = size + 4
    for offset, line in enumerate(lines):
        page.insert_text((x, y + offset * line_height), line, fontsize=size, fontname=fontname)
    return y + max(18, len(lines) * line_height + 6)


def _wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _para(seed: str) -> str:
    return " ".join([seed] * 4)
