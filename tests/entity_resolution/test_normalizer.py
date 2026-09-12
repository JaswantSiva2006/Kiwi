from __future__ import annotations

from kivi_memory.entity_resolution import normalize_entity_name


def test_normalize_entity_name_is_conservative() -> None:
    assert normalize_entity_name("  Priyaa,\tKapoor!!  ") == "priyaa kapoor"
    assert normalize_entity_name("Ａｔｌａｓ") == "atlas"
    assert normalize_entity_name("R&D + AI") == "r&d + ai"
