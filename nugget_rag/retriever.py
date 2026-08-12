"""Full-chunk and nugget retrieval modes for comparison."""
from __future__ import annotations

from nugget_rag.chunker import split_sentences
from nugget_rag.scorer import top_nuggets


def retrieve_full_chunk(chunks: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks as-is (baseline)."""
    return chunks[:top_k]


def retrieve_nuggets(
    chunks: list[dict],
    query: str,
    top_k: int = 5,
    nuggets_per_chunk: int = 3,
) -> list[dict]:
    """Extract top nugget sentences from each chunk, then return top-k.

    Args:
        chunks: List of chunk dicts with "text" field
        query: Query string
        top_k: Number of chunks to process
        nuggets_per_chunk: Number of sentences per chunk (default: 3)
    """
    results = []
    for chunk in chunks[:top_k]:
        sentences = split_sentences(chunk["text"])
        nugget_sents = top_nuggets(query, sentences, top_k=nuggets_per_chunk)
        results.append({**chunk, "nugget": " ".join(nugget_sents)})
    return results
