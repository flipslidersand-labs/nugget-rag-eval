"""Score sentences within a chunk against a query."""
from __future__ import annotations

import math
from collections import Counter


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


def top_nuggets(query: str, sentences: list[str], top_k: int = 2) -> list[str]:
    """Return top-k sentences most relevant to query (BM25)."""
    if not sentences:
        return []
    scores = bm25_scores(query, sentences)
    ranked = sorted(zip(scores, sentences), reverse=True)
    return [s for _, s in ranked[:top_k]]
