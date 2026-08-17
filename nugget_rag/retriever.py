"""Full-chunk and nugget retrieval modes for comparison."""

from __future__ import annotations

from collections.abc import Callable

from nugget_rag.chunker import split_sentences
from nugget_rag.scorer import bm25_scores, top_nuggets


def _rank_chunks(chunks: list[dict], query: str) -> list[dict]:
    """Rank chunks by BM25 score against query."""
    if not chunks:
        return []
    texts = [c.get("text", "") for c in chunks]
    scores = bm25_scores(query, texts)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked]


def retrieve_full_chunk(chunks: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks ranked by BM25 relevance to query."""
    return _rank_chunks(chunks, query)[:top_k]


def retrieve_nuggets(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
    nuggets_per_chunk: int = 3,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    embed_weight: float = 0.5,
) -> list[dict]:
    """Extract top nugget sentences from query-ranked chunks, then return top-k.

    When embed_fn is provided, nugget selection uses BM25 + embedding hybrid scoring.
    """
    results = []
    for chunk in _rank_chunks(chunks, query)[:top_k]:
        sentences = split_sentences(chunk["text"])
        nugget_sents = top_nuggets(
            query,
            sentences,
            top_k=nuggets_per_chunk,
            embed_fn=embed_fn,
            embed_weight=embed_weight,
        )
        results.append({**chunk, "nugget": " ".join(nugget_sents)})
    return results
