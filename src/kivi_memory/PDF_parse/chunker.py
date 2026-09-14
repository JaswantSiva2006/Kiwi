from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .schemas import SectionedBlock


TARGET_TOKENS = 600
MAX_TOKENS = 800
MIN_USEFUL_TOKENS = 150
OVERLAP_TOKENS = 100


def chunk_sections(sectioned_blocks: list[SectionedBlock], *, document_id: str, filename: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    grouped: dict[tuple[str | None, tuple[str, ...]], list[SectionedBlock]] = defaultdict(list)
    order: list[tuple[str | None, tuple[str, ...]]] = []
    for item in sectioned_blocks:
        key = (item.section_title, tuple(item.section_path))
        if key not in grouped:
            order.append(key)
        grouped[key].append(item)

    for key in order:
        _chunk_group(grouped[key], chunks, document_id=document_id, filename=filename)

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
        chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"] if index > 0 else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index < len(chunks) - 1 else None
    return chunks


def _chunk_group(items: list[SectionedBlock], chunks: list[dict[str, Any]], *, document_id: str, filename: str) -> None:
    current: list[SectionedBlock] = []
    current_tokens = 0
    for item in items:
        pieces = _split_oversized(item)
        for piece in pieces:
            tokens = _token_count(piece.block.text)
            should_flush = current and current_tokens + tokens > MAX_TOKENS
            if should_flush:
                _emit(current, chunks, document_id=document_id, filename=filename)
                current = _overlap_tail(current)
                current_tokens = sum(_token_count(x.block.text) for x in current)
            current.append(piece)
            current_tokens += tokens
            if current_tokens >= TARGET_TOKENS:
                _emit(current, chunks, document_id=document_id, filename=filename)
                current = _overlap_tail(current)
                current_tokens = sum(_token_count(x.block.text) for x in current)
    if current:
        if chunks and current_tokens < MIN_USEFUL_TOKENS and _same_section(chunks[-1], current[0]):
            _merge_into_previous(chunks[-1], current)
        else:
            _emit(current, chunks, document_id=document_id, filename=filename)


def _emit(items: list[SectionedBlock], chunks: list[dict[str, Any]], *, document_id: str, filename: str) -> None:
    text = "\n\n".join(item.block.text for item in items).strip()
    token_count = _token_count(text)
    chunk_index = len(chunks)
    block_ids = [item.block.block_id for item in items]
    page_start = min(item.block.page_number for item in items)
    page_end = max(item.block.page_number for item in items)
    chunk_type = _chunk_type(items)
    chunks.append(
        {
            "chunk_id": f"{document_id}:chunk:{chunk_index:05d}",
            "document_id": document_id,
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "text": text,
            "section_title": items[0].section_title,
            "section_path": items[0].section_path,
            "page_start": page_start,
            "page_end": page_end,
            "source_blocks": block_ids,
            "token_count": token_count,
            "previous_chunk_id": None,
            "next_chunk_id": None,
            "metadata": {"filename": filename, "mime_type": "application/pdf"},
        }
    )


def _merge_into_previous(chunk: dict[str, Any], items: list[SectionedBlock]) -> None:
    addition = "\n\n".join(item.block.text for item in items).strip()
    chunk["text"] = f"{chunk['text']}\n\n{addition}".strip()
    chunk["page_end"] = max(chunk["page_end"], max(item.block.page_number for item in items))
    chunk["source_blocks"].extend(item.block.block_id for item in items)
    chunk["token_count"] = _token_count(chunk["text"])


def _split_oversized(item: SectionedBlock) -> list[SectionedBlock]:
    if _token_count(item.block.text) <= MAX_TOKENS:
        return [item]
    sentences = re.split(r"(?<=[.!?])\s+", item.block.text)
    pieces: list[SectionedBlock] = []
    current: list[str] = []
    for sentence in sentences:
        if current and _token_count(" ".join(current + [sentence])) > MAX_TOKENS:
            pieces.append(_copy_with_text(item, " ".join(current)))
            current = []
        current.append(sentence)
    if current:
        pieces.append(_copy_with_text(item, " ".join(current)))
    return pieces


def _copy_with_text(item: SectionedBlock, text: str) -> SectionedBlock:
    block = item.block
    from .schemas import TextBlock

    copied = TextBlock(**{**block.__dict__, "text": text})
    return SectionedBlock(block=copied, section_title=item.section_title, section_path=item.section_path)


def _overlap_tail(items: list[SectionedBlock]) -> list[SectionedBlock]:
    tail: list[SectionedBlock] = []
    total = 0
    for item in reversed(items):
        tokens = _token_count(item.block.text)
        if tail and total + tokens > OVERLAP_TOKENS:
            break
        tail.insert(0, item)
        total += tokens
    return tail


def _same_section(chunk: dict[str, Any], item: SectionedBlock) -> bool:
    return chunk["section_title"] == item.section_title and chunk["section_path"] == item.section_path


def _chunk_type(items: list[SectionedBlock]) -> str:
    kinds = {item.block.kind for item in items}
    if kinds == {"CAPTION"}:
        return "CAPTION"
    return "TEXT"


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
