"""Score sentences within a chunk against a query."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Callable

from nugget_rag.vectors import embed_scores


def _char_ngrams(text: str, n: int = 2) -> list[str]:
    """Return character n-grams for a CJK run (whitespace already stripped).

    Uses ``"".join(text.split())`` so that full-width spaces, newlines, and
    tabs are all removed before n-gram generation — not just ASCII spaces.
    """
    text = "".join(text.split()).lower()
    return [text[i : i + n] for i in range(len(text) - n + 1)] if len(text) >= n else list(text)


def _tokenize(text: str) -> list[str]:
    """Hybrid tokenizer that handles mixed-language text uniformly.

    Strategy (applied to every text regardless of language ratio):
    - Latin/ASCII tokens are maximal runs of ASCII letters/digits in the
      lowercased text; punctuation ("accuracy.", "model,") is a separator.
    - Non-ASCII punctuation/symbols/spaces (Unicode categories P*/S*/Z*/C*,
      e.g. the apostrophe in "model’s") are separators too.
    - Consecutive remaining non-ASCII (CJK/Kana/etc.) runs are decomposed into character
      bigrams so that Japanese/Chinese terms can be matched even when the query
      is written in a different script.

    This ensures that an English query and a Japanese sentence share the same
    token space (e.g. "KV" matches "KV" inside "KVキャッシュ"), eliminating the
    structural mismatch that caused BM25 to return all-zero scores for
    cross-language pairs.
    """
    tokens: list[str] = []
    ascii_buf: list[str] = []
    cjk_run: list[str] = []

    def _flush() -> None:
        if ascii_buf:
            tokens.append("".join(ascii_buf))
            ascii_buf.clear()
        if cjk_run:
            tokens.extend(_char_ngrams("".join(cjk_run)))
            cjk_run.clear()

    for ch in text.lower():
        if ch.isascii():
            if ch.isalnum():
                if cjk_run:
                    _flush()
                ascii_buf.append(ch)
            else:  # whitespace and ASCII punctuation separate tokens
                _flush()
        elif ch.isspace() or unicodedata.category(ch)[0] in "PSZC":
            _flush()  # non-ASCII punctuation/symbols/spaces: separator, not a CJK char
        else:
            if ascii_buf:
                _flush()
            cjk_run.append(ch)
    _flush()

    return tokens


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
    # Pre-compute document frequency (df) for each query term:
    # df[term] = number of documents in the corpus that contain term.
    # This must be computed outside the per-document loop so that IDF
    # reflects corpus-level rarity, not the current document's TF.
    N = len(sentences)
    df: dict[str, int] = {term: sum(1 for doc in tokenized if term in doc) for term in set(q_terms)}
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)  # within-document term frequency (TF)
            n = df[term]  # corpus document frequency (DF) used for IDF
            idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. Returns zeros if all scores equal."""
    if not scores:
        return []
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
    query_vec: list[float] | None = None,
    sent_vecs: list[list[float]] | None = None,
    window: int = 0,
) -> list[str]:
    """Return top-k sentences most relevant to query.

    ``window`` adds that many neighbouring sentences on each side of every
    selected sentence (0 = off, behaviour unchanged). Output stays in document order.

    When embed_fn is provided, scores are a weighted blend of BM25 and
    cosine similarity (embed_weight controls the embedding contribution).
    Without embed_fn, falls back to BM25-only.

    Pre-computed vectors can be supplied via ``query_vec`` and ``sent_vecs``
    to avoid redundant HTTP calls when batching across multiple chunks.
    When both are provided, ``embed_fn`` is not called.
    Providing only one of ``query_vec`` / ``sent_vecs`` raises ValueError.
    """
    if not sentences:
        return []

    bm25 = _normalize(bm25_scores(query, sentences))

    if (query_vec is None) != (sent_vecs is None):
        raise ValueError("query_vec and sent_vecs must be provided together")
    use_precomputed = query_vec is not None and sent_vecs is not None
    if use_precomputed or embed_fn is not None:
        if use_precomputed:
            qv = query_vec
            sv = sent_vecs
        else:
            vecs = embed_fn([query] + sentences)  # type: ignore[misc]
            if len(vecs) != 1 + len(sentences):
                raise ValueError(
                    f"embed_fn returned {len(vecs)} vectors, expected {1 + len(sentences)}"
                )
            qv, sv = vecs[0], vecs[1:]
        if len(sv) != len(sentences):
            raise ValueError(f"sent_vecs has {len(sv)} vectors for {len(sentences)} sentences")
        emb = _normalize(embed_scores(qv, sv))
        combined = [(1 - embed_weight) * b + embed_weight * e for b, e in zip(bm25, emb)]
    else:
        combined = bm25

    # Select top-k indices by score
    top_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]
    # Sort indices back to original order to preserve context flow
    selected = set(top_indices)
    for i in top_indices:
        selected.update(j for j in range(i - window, i + window + 1) if 0 <= j < len(sentences))
    return [sentences[i] for i in sorted(selected)]
