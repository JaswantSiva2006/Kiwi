from __future__ import annotations

from collections import Counter
from typing import Any


def validate_output(output: dict[str, Any], block_ids: set[str]) -> list[str]:
    warnings: list[str] = []
    document = output["document"]
    chunks = output["chunks"]
    page_count = document["page_count"]

    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        warnings.append("duplicate_chunk_ids")

    empty = [chunk["chunk_id"] for chunk in chunks if not chunk["text"].strip()]
    if empty:
        warnings.append(f"empty_chunk_text:{len(empty)}")

    for index, chunk in enumerate(chunks):
        if chunk["chunk_index"] != index:
            warnings.append("chunk_ordering")
        if chunk["page_start"] < 1 or chunk["page_end"] < 1 or chunk["page_end"] > page_count:
            warnings.append(f"invalid_page_range:{chunk['chunk_id']}")
        if chunk["page_start"] > chunk["page_end"]:
            warnings.append(f"page_start_after_end:{chunk['chunk_id']}")
        if chunk["token_count"] <= 0:
            warnings.append(f"non_positive_token_count:{chunk['chunk_id']}")
        if chunk["token_count"] > 1000:
            warnings.append(f"absurdly_oversized_chunk:{chunk['chunk_id']}")
        missing = [block_id for block_id in chunk["source_blocks"] if block_id not in block_ids]
        if missing:
            warnings.append(f"missing_source_blocks:{chunk['chunk_id']}")

    normalized = [" ".join(chunk["text"].lower().split()) for chunk in chunks]
    if normalized:
        duplicate_count = sum(count - 1 for count in Counter(normalized).values() if count > 1)
        ratio = duplicate_count / len(normalized)
        if ratio > 0.2:
            warnings.append(f"high_duplicate_chunk_ratio:{ratio:.2f}")

    pages_with_text = {page for chunk in chunks for page in range(chunk["page_start"], chunk["page_end"] + 1)}
    if page_count:
        empty_page_ratio = (page_count - len(pages_with_text)) / page_count
        if empty_page_ratio > 0.35:
            warnings.append(f"high_empty_page_ratio:{empty_page_ratio:.2f}")
    return sorted(set(warnings))
