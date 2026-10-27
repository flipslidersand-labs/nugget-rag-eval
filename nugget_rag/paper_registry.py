"""Stable mapping between arxiv_id and internal paper_id.

paper_id values are assigned by the academic-paper-system DB and may change
on DB rebuild; arxiv_id is the stable external identifier.
"""

from __future__ import annotations

# arxiv_id → internal paper_id
ARXIV_MAP: dict[str, int] = {
    "2410.10071": 1,
    "2508.11836": 2,
    "2508.11845": 3,
    "2509.25673": 4,
    "2511.07482": 5,
    "2512.06812": 6,
    "2601.10849": 7,
    "2602.10161": 8,
    "2608.06495": 9,
    "2608.07458": 10,
}

# Reverse: paper_id → arxiv_id
PAPER_ID_TO_ARXIV: dict[int, str] = {v: k for k, v in ARXIV_MAP.items()}


def resolve_paper_key(item: dict) -> str | int | None:
    """Resolve a chunk or gold item to its canonical registry key.

    Prefers ``arxiv_id`` translated through ARXIV_MAP to the internal
    ``paper_id`` so that chunks and gold items sharing the same arxiv_id
    always resolve to the same key regardless of which (possibly stale)
    ``paper_id`` they carry. Falls back to the raw arxiv_id string if it is
    not registered, then to the ``paper_id`` field. Returns ``None`` if
    neither is present.
    """
    arxiv_id = item.get("arxiv_id")
    if arxiv_id:
        return ARXIV_MAP.get(arxiv_id, arxiv_id)
    return item.get("paper_id")
