"""Pure vector math shared by scoring (no I/O, stdlib only).

Lives apart from embedder.py (the HTTP client) so that scorer.py depends on the
algorithm layer only; swapping the embedding backend never touches scorer.py (#185).
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_scores(query_vec: list[float], sentence_vecs: list[list[float]]) -> list[float]:
    """Return cosine similarity of each sentence vector against the query vector."""
    return [cosine_similarity(query_vec, sv) for sv in sentence_vecs]
