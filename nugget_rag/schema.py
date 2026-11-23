"""Chunk dict schema and the single place that builds it."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from nugget_rag.paper_registry import PAPER_ID_TO_ARXIV


class Chunk(TypedDict):
    """Pool chunk. Exactly one of chunk_index / chunk_indices is set by make_chunk."""

    paper_id: int
    text: str
    score: float
    arxiv_id: NotRequired[str]
    chunk_index: NotRequired[int]
    chunk_indices: NotRequired[list[int]]


def make_chunk(
    paper_id: int,
    text: str,
    score: float,
    *,
    chunk_index: int | None = None,
    chunk_indices: list[int] | None = None,
    arxiv_id: str | None = None,
    arxiv_id_first: bool = False,
) -> Chunk:
    """Build a chunk dict.

    arxiv_id defaults to PAPER_ID_TO_ARXIV[paper_id] and is omitted if unknown; pass it
    explicitly when the caller already knows it (the PDF pool takes it from the file name).

    Pass chunk_index (small mode) or chunk_indices (large / PDF mode), not both. Key order is
    paper_id, <index key>, text, score, arxiv_id so that the output JSON stays byte-identical;
    arxiv_id_first=True puts arxiv_id right after paper_id (the PDF pool's existing layout).
    Unifying the index key and the key order is a separate change.
    """
    if (chunk_index is None) == (chunk_indices is None):
        raise ValueError("exactly one of chunk_index / chunk_indices is required")
    index = (
        {"chunk_index": chunk_index} if chunk_indices is None else {"chunk_indices": chunk_indices}
    )
    if arxiv_id is None:
        arxiv_id = PAPER_ID_TO_ARXIV.get(paper_id)
    arxiv = {} if arxiv_id is None else {"arxiv_id": arxiv_id}
    if arxiv_id_first:
        return {"paper_id": paper_id, **arxiv, **index, "text": text, "score": score}  # type: ignore[typeddict-item]
    return {"paper_id": paper_id, **index, "text": text, "score": score, **arxiv}  # type: ignore[typeddict-item]
