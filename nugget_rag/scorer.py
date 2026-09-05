"""Score sentences within a chunk against a query."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable


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
    - Latin/ASCII words are extracted by whitespace-splitting the lowercased text.
    - Consecutive non-ASCII (CJK/Kana/etc.) runs are decomposed into character
      bigrams so that Japanese/Chinese terms can be matched even when the query
      is written in a different script.

    This ensures that an English query and a Japanese sentence share the same
    token space (e.g. "KV" matches "KV" inside "KVキャッシュ"), eliminating the
    structural mismatch that caused BM25 to return all-zero scores for
    cross-language pairs.
    """
    tokens: list[str] = []
    text_lower = text.lower()
    # Accumulate the current non-ASCII run to bigram-ise later.
    cjk_run: list[str] = []

    def _flush_cjk() -> None:
        if not cjk_run:
            return
        run = "".join(cjk_run)
        tokens.extend(_char_ngrams(run))
        cjk_run.clear()

    for word in text_lower.split():
        # Split each whitespace-delimited word into ASCII and non-ASCII parts.
        ascii_buf: list[str] = []
        for ch in word:
            if ch.isascii():
                if cjk_run:
                    _flush_cjk()
                ascii_buf.append(ch)
            else:
                if ascii_buf:
                    # Emit the accumulated ASCII fragment as one token.
                    fragment = "".join(ascii_buf).strip()
                    if fragment:
                        tokens.append(fragment)
                    ascii_buf.clear()
                cjk_run.append(ch)
        if ascii_buf:
            fragment = "".join(ascii_buf).strip()
            if fragment:
                tokens.append(fragment)
        if cjk_run:
            _flush_cjk()

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
) -> list[str]:
    """Return top-k sentences most relevant to query.

    When embed_fn is provided, scores are a weighted blend of BM25 and
    cosine similarity (embed_weight controls the embedding contribution).
    Without embed_fn, falls back to BM25-only.

    Pre-computed vectors can be supplied via ``query_vec`` and ``sent_vecs``
    to avoid redundant HTTP calls when batching across multiple chunks.
    When both are provided, ``embed_fn`` is not called.
    """
    if not sentences:
        return []

    bm25 = _normalize(bm25_scores(query, sentences))

    use_precomputed = query_vec is not None and sent_vecs is not None
    if use_precomputed or embed_fn is not None:
        from nugget_rag.embedder import embed_scores

        if use_precomputed:
            qv = query_vec
            sv = sent_vecs
        else:
            vecs = embed_fn([query] + sentences)  # type: ignore[misc]
            qv, sv = vecs[0], vecs[1:]
        emb = _normalize(embed_scores(qv, sv))
        combined = [(1 - embed_weight) * b + embed_weight * e for b, e in zip(bm25, emb)]
    else:
        combined = bm25

    # Select top-k indices by score
    top_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]
    # Sort indices back to original order to preserve context flow
    top_indices_sorted = sorted(top_indices)
    return [sentences[i] for i in top_indices_sorted]
