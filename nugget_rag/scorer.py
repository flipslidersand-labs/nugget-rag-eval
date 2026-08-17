"""Score sentences within a chunk against a query."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_scores(
    query: str,
    sentences: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Return BM25 score for each sentence relative to the query."""
    if not sentences:
        return []
    tokenized = [_tokenize(s) for s in sentences]
    avgdl = sum(len(t) for t in tokenized) / len(tokenized)
    q_terms = _tokenize(query)
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            idf = math.log(1 + (len(sentences) - f + 0.5) / (f + 0.5))
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. Returns zeros if all scores equal."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def top_nuggets(
    query: str,
    sentences: list[str],
    top_k: int = 2,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    embed_weight: float = 0.5,
) -> list[str]:
    """Return top-k sentences most relevant to query.

    When embed_fn is provided, scores are a weighted blend of BM25 and
    cosine similarity (embed_weight controls the embedding contribution).
    Without embed_fn, falls back to BM25-only.
    """
    if not sentences:
        return []

    bm25 = _normalize(bm25_scores(query, sentences))

    if embed_fn is not None:
        from nugget_rag.embedder import embed_scores

        vecs = embed_fn([query] + sentences)
        query_vec, sent_vecs = vecs[0], vecs[1:]
        emb = _normalize(embed_scores(query_vec, sent_vecs))
        combined = [(1 - embed_weight) * b + embed_weight * e for b, e in zip(bm25, emb)]
    else:
        combined = bm25

    ranked = sorted(zip(combined, sentences), reverse=True)
    return [s for _, s in ranked[:top_k]]
