"""Span matching and metrics (Recall@k, MRR@k, token counts, percentiles).

Pure functions shared by evaluate / check_regression / scripts. Imports nothing from
``eval.evaluate`` (the dependency goes the other way).
"""

from __future__ import annotations

import re
import unicodedata

_NON_WORD = re.compile(r"[\W_]+")


def _normalize_for_match(text: str) -> str:
    """NFKC-fold, lowercase and drop whitespace/punctuation (keeps letters and digits, incl. CJK).

    Text extractors differ in spacing ("University of California" vs
    "UniversityofCalifornia") and in ligatures / full-width forms ("eﬃcient",
    "９５％"), so spans taken from one extraction only match another after this
    normalization.
    """
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", text).lower())


def span_in(text: str, spans: list[str], normalize: bool) -> bool:
    if normalize:
        norm_text = _normalize_for_match(text)
        return any(n and n in norm_text for n in map(_normalize_for_match, spans))
    low = text.lower()
    return any(span.strip() and span.lower() in low for span in spans)


def recall_at_k(
    results: list[dict], answer_spans: list[str], field: str = "text", normalize: bool = False
) -> bool:
    """Return True if any single result contains any answer span.

    Judged per-result (same criterion as :func:`mrr_at_k`) — never on joined
    text, which would create false positives when a span only matches across
    the boundary of two concatenated results (#146).
    """
    return any(span_in(r.get(field, ""), answer_spans, normalize) for r in results)


def mrr_at_k(
    results: list[dict], answer_spans: list[str], field: str = "text", normalize: bool = False
) -> float:
    """Return reciprocal rank of the first hit (1-indexed), 0 if no hit."""
    for rank, result in enumerate(results, start=1):
        if span_in(result.get(field, ""), answer_spans, normalize):
            return 1.0 / rank
    return 0.0


def check_estimator(estimator: str) -> None:
    if estimator not in ("words", "chars"):
        raise ValueError(f"estimator must be 'words' or 'chars', got {estimator!r}")


def avg_tokens(results: list[dict], field: str = "text", estimator: str = "words") -> float:
    check_estimator(estimator)
    texts = [r.get(field, "") for r in results]
    if estimator == "chars":
        return sum(len(t) / 4 for t in texts) / max(len(texts), 1)
    return sum(len(t.split()) for t in texts) / max(len(texts), 1)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0-100). Empty input -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil without math import
    return ordered[int(rank) - 1]
