from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import median

from .schemas import TextBlock


NUMBERED_HEADING_RE = re.compile(
    r"^\s*((\d+(\.\d+){0,5}\.?)|((appendix|chapter|section)\s+[A-Z0-9]+))(\s+|:|$)",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table|algorithm|alg\.)\s+\d+[\.:]", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*([\-*•]|\(?[a-zA-Z0-9]{1,3}\)|\d+[.)])\s+")


def detect_headings(blocks: list[TextBlock]) -> list[dict[str, object]]:
    if not blocks:
        return []
    body_size = _dominant_body_size(blocks)
    repeated_edges = _repeated_page_edge_text(blocks)
    diagnostics: list[dict[str, object]] = []

    for block in blocks:
        score, signals, strong = _score_block(block, body_size, repeated_edges)
        block.heading_score = score
        block.heading_signals = signals
        if CAPTION_RE.match(block.text):
            block.kind = "CAPTION"
        elif BULLET_RE.match(block.text):
            block.kind = "LIST"
        elif score >= 4 and strong:
            block.kind = "HEADING"
        else:
            block.kind = "BODY"
        diagnostics.append(
            {
                "text": block.text,
                "heading_score": score,
                "signals": signals,
                "classified_as": block.kind if block.kind in {"HEADING", "BODY"} else "OTHER",
            }
        )
    return diagnostics


def _dominant_body_size(blocks: list[TextBlock]) -> float:
    candidates = [b.font_size for b in blocks if len(b.text.split()) >= 8]
    if not candidates:
        candidates = [b.font_size for b in blocks]
    rounded = [round(size * 2) / 2 for size in candidates if size > 0]
    if not rounded:
        return 12.0
    return Counter(rounded).most_common(1)[0][0]


def _repeated_page_edge_text(blocks: list[TextBlock]) -> set[str]:
    pages: dict[int, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        pages[block.page_number].append(block)
    repeated: Counter[str] = Counter()
    for page_blocks in pages.values():
        if not page_blocks:
            continue
        heights = [b.bbox[3] for b in page_blocks]
        page_bottom = max(heights)
        for block in page_blocks:
            near_edge = block.bbox[1] < 72 or block.bbox[3] > page_bottom - 72
            if near_edge and len(block.text) < 120:
                repeated[_canonical(block.text)] += 1
    return {text for text, count in repeated.items() if count >= 2}


def _score_block(block: TextBlock, body_size: float, repeated_edges: set[str]) -> tuple[int, list[str], bool]:
    text = block.text.strip()
    words = text.split()
    score = 0
    signals: list[str] = []
    strong = False

    if block.font_size >= body_size + 1.75:
        score += 3
        signals.append("larger_font")
        strong = True
    if _looks_bold(block):
        score += 2
        signals.append("bold")
        strong = True
    if NUMBERED_HEADING_RE.match(text):
        score += 2
        signals.append("numbered_section")
        strong = True
    if 1 <= len(words) <= 12 and len(text) <= 100:
        score += 1
        signals.append("short_block")
    if block.gap_before >= max(10.0, block.font_size * 1.4):
        score += 1
        signals.append("large_gap_before")
    if _heading_caps(text):
        score += 1
        signals.append("heading_capitalization")

    if len(words) >= 35 or len(text) >= 240:
        score -= 3
        signals.append("long_paragraph")
    if CAPTION_RE.match(text):
        score -= 2
        signals.append("caption")
    if BULLET_RE.match(text):
        score -= 2
        signals.append("bullet_or_list")
    if _canonical(text) in repeated_edges:
        score -= 2
        signals.append("repeated_header_footer")
    return score, signals, strong


def _looks_bold(block: TextBlock) -> bool:
    fonts = " ".join(block.fonts).lower()
    return bool(block.flags & 16) or "bold" in fonts or "semibold" in fonts or "demi" in fonts


def _heading_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    words = [w.strip(".,:;()[]") for w in text.split()]
    title_like = sum(1 for w in words if w[:1].isupper())
    return text.isupper() or title_like >= max(1, int(len(words) * 0.65))


def _canonical(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))
