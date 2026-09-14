from __future__ import annotations

import re

from .schemas import SectionedBlock, TextBlock


NUMBER_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\.?\s+(.+)$")
APPENDIX_RE = re.compile(r"^\s*appendix\s+([A-Z0-9]+)\b[:\s-]*(.*)$", re.IGNORECASE)


def build_sections(blocks: list[TextBlock]) -> list[SectionedBlock]:
    section_stack: list[tuple[int, str, float]] = []
    sectioned: list[SectionedBlock] = []
    font_levels: list[float] = []

    for block in blocks:
        if block.kind == "HEADING":
            depth = _numbering_depth(block.text)
            if depth is None:
                depth = _font_depth(block.font_size, font_levels)
            title = _clean_heading_title(block.text)
            section_stack = [item for item in section_stack if item[0] < depth]
            section_stack.append((depth, title, block.font_size))
            continue

        path = [title for _, title, _ in sorted(section_stack, key=lambda item: item[0])]
        sectioned.append(
            SectionedBlock(
                block=block,
                section_title=path[-1] if path else None,
                section_path=path,
            )
        )
    return sectioned


def _numbering_depth(text: str) -> int | None:
    match = NUMBER_PREFIX_RE.match(text)
    if match:
        return match.group(1).count(".") + 1
    if APPENDIX_RE.match(text):
        return 1
    return None


def _font_depth(font_size: float, font_levels: list[float]) -> int:
    for index, size in enumerate(font_levels, start=1):
        if abs(font_size - size) <= 0.5:
            return index
    font_levels.append(font_size)
    font_levels.sort(reverse=True)
    return font_levels.index(font_size) + 1


def _clean_heading_title(text: str) -> str:
    return " ".join(text.split())
