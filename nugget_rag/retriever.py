"""Full-chunk and nugget retrieval modes for comparison."""

from __future__ import annotations

from collections.abc import Callable

from nugget_rag.chunker import split_sentences
from nugget_rag.scorer import bm25_scores, top_nugget_indices

# Marks a gap between non-adjacent picked sentences (#301). It must be a word character:
# the normalized match mode drops whitespace/punctuation, so any other separator would
# vanish and the two sentences would still match across the gap. U+01C0 is a letter
# (survives NFKC + lower) and absent from the data. It is glued to the next sentence
# (" " + marker + sentence) so the whitespace word count is the same as a plain join.
_SEAM = "\u01c0"
# Sentence ends after which split_sentences() removed no whitespace (see chunker._SENT_RE).
_CJK_ENDS = "。！？"


def _join_nuggets(sentences: list[str], indices: list[int]) -> str:
    """Join picked sentences without creating text that is absent from the chunk (#301).

    Consecutive sentences are joined as in the source: no space after a CJK sentence end,
    one space otherwise. A gap between non-adjacent picks gets ``" " + _SEAM`` so no span
    can match across it.
    """
    out = ""
    prev: int | None = None
    for i in indices:
        if prev is None:
            out = sentences[i]
        elif i == prev + 1:
            out += ("" if sentences[prev].endswith(tuple(_CJK_ENDS)) else " ") + sentences[i]
        else:
            out += f" {_SEAM}{sentences[i]}"
        prev = i
    return out


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


def _embed_batch(
    embed_fn: Callable[[list[str]], list[list[float]]], query: str, sentences: list[str]
) -> tuple[list[float], list[list[float]]]:
    """Embed query + sentences in one call; return (query_vec, sentence_vecs) (#150, #251)."""
    vecs = embed_fn([query] + sentences)
    if len(vecs) != 1 + len(sentences):
        raise ValueError(f"embed_fn returned {len(vecs)} vectors, expected {1 + len(sentences)}")
    return vecs[0], vecs[1:]


def retrieve_nuggets(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
    nuggets_per_chunk: int = 3,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    embed_weight: float = 0.5,
    window: int = 0,
) -> list[dict]:
    """Extract top nugget sentences from query-ranked chunks, then return top-k.

    ``window`` > 0 also returns that many neighbouring sentences around each pick (#201).

    When embed_fn is provided and embed_weight > 0, nugget selection uses BM25 + embedding
    hybrid scoring. embed_weight == 0 is BM25-only and never calls embed_fn (#239).

    Batching optimisation (#150): when embed_fn is provided, all sentences
    from the top-k chunks are collected into a single embed_fn call together
    with the query.  This reduces HTTP round-trips from O(top_k) to O(1)
    per query and eliminates repeated query-vector computation.
    """
    ranked_chunks = _rank_chunks(chunks, query)[:top_k]
    chunk_sentences: list[list[str]] = [split_sentences(c["text"]) for c in ranked_chunks]

    # weight 0 -> embedding adds nothing to the score; skip the service call (#239)
    query_vec: list[float] | None = None
    flat_sent_vecs: list[list[float]] | None = None
    all_sentences = [s for sents in chunk_sentences for s in sents]
    if embed_fn is not None and embed_weight != 0 and all_sentences:
        query_vec, flat_sent_vecs = _embed_batch(embed_fn, query, all_sentences)

    results = []
    offset = 0
    for chunk, sentences in zip(ranked_chunks, chunk_sentences):
        n = len(sentences)
        sent_vecs = None
        if flat_sent_vecs is not None:
            sent_vecs = flat_sent_vecs[offset : offset + n]
            offset += n
        nugget_idx = top_nugget_indices(
            query,
            sentences,
            top_k=nuggets_per_chunk,
            embed_weight=embed_weight,
            query_vec=query_vec,
            sent_vecs=sent_vecs,
            window=window,
        )
        results.append({**chunk, "nugget": _join_nuggets(sentences, nugget_idx)})
    return results
