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

    Batching optimisation (#150): when embed_fn is provided, all sentences
    from the top-k chunks are collected into a single embed_fn call together
    with the query.  This reduces HTTP round-trips from O(top_k) to O(1)
    per query and eliminates repeated query-vector computation.
    """
    ranked_chunks = _rank_chunks(chunks, query)[:top_k]

    if embed_fn is None:
        # BM25-only path — no batching needed
        results = []
        for chunk in ranked_chunks:
            sentences = split_sentences(chunk["text"])
            nugget_sents = top_nuggets(
                query,
                sentences,
                top_k=nuggets_per_chunk,
                embed_weight=embed_weight,
            )
            results.append({**chunk, "nugget": " ".join(nugget_sents)})
        return results

    # Hybrid path — one batch embed call for query + all chunk sentences
    chunk_sentences: list[list[str]] = [split_sentences(c["text"]) for c in ranked_chunks]
    all_sentences: list[str] = [s for sents in chunk_sentences for s in sents]

    if all_sentences:
        all_vecs = embed_fn([query] + all_sentences)
        query_vec: list[float] = all_vecs[0]
        flat_sent_vecs: list[list[float]] = all_vecs[1:]
    else:
        query_vec = []
        flat_sent_vecs = []

    results = []
    offset = 0
    for chunk, sentences in zip(ranked_chunks, chunk_sentences):
        n = len(sentences)
        sent_vecs = flat_sent_vecs[offset : offset + n]
        offset += n
        nugget_sents = top_nuggets(
            query,
            sentences,
            top_k=nuggets_per_chunk,
            embed_weight=embed_weight,
            query_vec=query_vec if sentences else None,
            sent_vecs=sent_vecs if sentences else None,
        )
        results.append({**chunk, "nugget": " ".join(nugget_sents)})
    return results
