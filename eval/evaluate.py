"""Evaluate full-chunk vs nugget retrieval using Recall@k and token count.

Usage:
    # BM25-only (default)
    python eval/evaluate.py --chunks data/chunks.json --gold eval/gold_set.json

    # BM25 + embedding hybrid nugget scoring
    # Set EMBEDDING_API_KEY env var (do NOT pass via CLI — it would appear in ps/logs)
    EMBEDDING_API_KEY=<key> python eval/evaluate.py \
        --chunks data/chunks_large_perquery.json \
        --gold eval/gold_set.json \
        --embedding-url http://<internal-host>:9092 \
        --embed-weight 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from eval.cli_helpers import existing_file as _existing_file
from eval.cli_helpers import float_between_0_1 as _float_between_0_1
from eval.cli_helpers import load_json as _load_json
from eval.cli_helpers import non_negative_int as _non_negative_int
from eval.cli_helpers import positive_int as _positive_int
from eval.compare import compare
from nugget_rag.paper_registry import resolve_paper_key
from nugget_rag.retriever import retrieve_full_chunk, retrieve_nuggets

_REQUIRED_FIELDS = frozenset({"query", "answer_spans"})


def validate_chunks(chunks: list[dict]) -> None:
    """Validate chunks list, raising ValueError on schema violations.

    Checks:
    - ``chunks`` is a list (not a dict or other type).
    - Each chunk has at least one of ``arxiv_id`` or ``paper_id``.
    - Each chunk has a non-empty ``text`` field.

    Raises:
        ValueError: On the first violation, with a descriptive message listing
            the index and the specific problem.
    """
    if not isinstance(chunks, list):
        raise ValueError(f"chunks must be a JSON array (list), got {type(chunks).__name__!r}")
    for i, chunk in enumerate(chunks):
        has_id = chunk.get("arxiv_id") or chunk.get("paper_id") is not None
        if not has_id:
            raise ValueError(
                f"chunks[{i}] must have 'arxiv_id' or 'paper_id' (both absent or falsy)"
            )
        text = chunk.get("text")
        if not text:
            raise ValueError(
                f"chunks[{i}] missing or empty 'text' field — recall would silently drop to 0"
            )


def build_chunks_index(chunks: list[dict]) -> dict[str | int, list[dict]]:
    """Group a flat chunks list into a dict keyed by paper identifier.

    Uses :func:`resolve_paper_key` so chunks resolve to the same key as gold
    items sharing the same arxiv_id, regardless of which paper_id they carry.
    Assumes *chunks* has already been validated by :func:`validate_chunks`.
    """
    chunks_by_paper: dict[str | int, list[dict]] = {}
    for c in chunks:
        key = resolve_paper_key(c)
        chunks_by_paper.setdefault(key, []).append(c)
    return chunks_by_paper


def validate_gold(gold: list[dict]) -> None:
    """Validate gold set items, raising ValueError on schema violations.

    Checks:
    - Each item has at least one of ``arxiv_id`` or ``paper_id``.
    - Each item has ``query`` and ``answer_spans``.
    - ``answer_spans`` is a non-empty list of strings.

    Raises:
        ValueError: On the first item that fails validation, with a descriptive
            message listing the index and the specific problem.
    """
    for i, item in enumerate(gold):
        missing = _REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(f"gold[{i}] missing required fields: {sorted(missing)}")

        has_id = item.get("arxiv_id") or ("paper_id" in item)
        if not has_id:
            raise ValueError(f"gold[{i}] must have 'arxiv_id' or 'paper_id' (both absent or falsy)")

        spans = item["answer_spans"]
        if not isinstance(spans, list):
            raise ValueError(
                f"gold[{i}] 'answer_spans' must be a list, got {type(spans).__name__!r}"
            )
        if len(spans) == 0:
            raise ValueError(
                f"gold[{i}] 'answer_spans' is empty — recall would always be 0 (silent error)"
            )
        non_str = [s for s in spans if not isinstance(s, str)]
        if non_str:
            raise ValueError(
                f"gold[{i}] 'answer_spans' contains non-string elements: {non_str[:3]}"
            )


def check_gold_chunk_coverage(chunks_by_paper: dict[str, list[dict]], gold: list[dict]) -> int:
    """Warn about gold keys missing from chunks_by_paper; return the count of affected queries.

    Prints a ``[WARN]`` line to stderr with the count and up to 5 example keys.

    Raises:
        ValueError: If *gold* is non-empty and no gold key resolves to any chunks
            (every query would silently score recall 0).
    """
    missing = [key for item in gold if (key := resolve_paper_key(item)) not in chunks_by_paper]
    if not missing:
        return 0

    examples = sorted({str(k) for k in missing})[:5]
    if len(missing) == len(gold):
        raise ValueError(
            f"no gold paper key found in chunks — all {len(gold)} queries would score "
            f"recall 0 (wrong --chunks file or unregistered arxiv_id?); examples: {examples}"
        )
    print(
        f"[WARN] {len(missing)}/{len(gold)} gold queries have no chunks "
        f"(keys not in chunks data): {examples}",
        file=sys.stderr,
    )
    return len(missing)


_NON_WORD = re.compile(r"[\W_]+")


def _normalize_for_match(text: str) -> str:
    """Lowercase and drop whitespace/punctuation (keeps letters and digits, incl. CJK).

    Text extractors differ in spacing ("University of California" vs
    "UniversityofCalifornia"), so spans taken from one extraction only match
    another after this normalization.
    """
    return _NON_WORD.sub("", text.lower())


def _span_in(text: str, spans: list[str], normalize: bool) -> bool:
    if normalize:
        norm_text = _normalize_for_match(text)
        return any(n and n in norm_text for n in map(_normalize_for_match, spans))
    low = text.lower()
    return any(span.lower() in low for span in spans)


def recall_at_k(
    results: list[dict], answer_spans: list[str], field: str = "text", normalize: bool = False
) -> bool:
    """Return True if any single result contains any answer span.

    Judged per-result (same criterion as :func:`mrr_at_k`) — never on joined
    text, which would create false positives when a span only matches across
    the boundary of two concatenated results (#146).
    """
    return any(_span_in(r.get(field, ""), answer_spans, normalize) for r in results)


def mrr_at_k(
    results: list[dict], answer_spans: list[str], field: str = "text", normalize: bool = False
) -> float:
    """Return reciprocal rank of the first hit (1-indexed), 0 if no hit."""
    for rank, result in enumerate(results, start=1):
        if _span_in(result.get(field, ""), answer_spans, normalize):
            return 1.0 / rank
    return 0.0


def avg_tokens(results: list[dict], field: str = "text", estimator: str = "words") -> float:
    texts = [r.get(field, "") for r in results]
    if estimator == "chars":
        return sum(len(t) / 4 for t in texts) / max(len(texts), 1)
    return sum(len(t.split()) for t in texts) / max(len(texts), 1)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0-100). Empty input -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil without math import
    return ordered[int(rank) - 1]


def _latency_summary(values_s: list[float]) -> dict:
    return {
        "p50": round(_percentile(values_s, 50) * 1000, 2),
        "p95": round(_percentile(values_s, 95) * 1000, 2),
    }


@dataclass(frozen=True)
class RetrievalMode:
    """One retrieval mode to evaluate. Add an entry to MODES to add a mode."""

    name: str  # key in the result dict (and in baseline files)
    field: str  # chunk field holding the text that is judged and counted
    label: str  # display name
    show_misses: bool = False  # list recall misses under --verbose


MODES: tuple[RetrievalMode, ...] = (
    RetrievalMode("full_chunk", "text", "full-chunk"),
    RetrievalMode("nugget", "nugget", "nugget", show_misses=True),
)


def mode_label(name: str, embed_weight: float | None = None) -> str:
    """Display label for a mode; nugget shows the embed weight when embeddings are used."""
    label = next(m.label for m in MODES if m.name == name)
    if name == "nugget" and embed_weight is not None:
        return f"nugget(e{embed_weight:.1f})"
    return label


def evaluate(
    chunks_by_paper: dict[str, list[dict]],
    gold: list[dict],
    top_k: int = 5,
    verbose: bool = False,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    embed_weight: float = 0.5,
    estimator: str = "words",
    nugget_window: int = 0,
    match: str = "exact",
) -> dict:
    if match not in ("exact", "normalized"):
        raise ValueError(f"match must be 'exact' or 'normalized', got {match!r}")
    normalize = match == "normalized"
    validate_gold(gold)
    n_queries_without_chunks = check_gold_chunk_coverage(chunks_by_paper, gold)

    embed_spent = 0.0
    timed_embed_fn = embed_fn
    if embed_fn is not None:

        def timed_embed_fn(texts):  # noqa: F811 - wraps embed_fn to time HTTP calls
            nonlocal embed_spent
            t0 = time.perf_counter()
            try:
                return embed_fn(texts)
            finally:
                embed_spent += time.perf_counter() - t0

    retrievers: dict[str, Callable[[list[dict], str], list[dict]]] = {
        "full_chunk": lambda chunks, query: retrieve_full_chunk(chunks, query, top_k),
        "nugget": lambda chunks, query: retrieve_nuggets(
            chunks,
            query,
            top_k,
            embed_fn=timed_embed_fn,
            embed_weight=embed_weight,
            window=nugget_window,
        ),
    }
    assert set(retrievers) == {m.name for m in MODES}, "MODES and retrievers out of sync"

    hits = {m.name: 0 for m in MODES}
    tokens = {m.name: 0.0 for m in MODES}
    mrr_sum = {m.name: 0.0 for m in MODES}
    times: dict[str, list[float]] = {m.name: [] for m in MODES}
    flags: dict[str, list[bool]] = {m.name: [] for m in MODES}
    misses: dict[str, list[tuple[int, str, str]]] = {m.name: [] for m in MODES}
    embed_times: list[float] = []

    for i, item in enumerate(gold):
        # Support both arxiv_id (new) and paper_id (legacy)
        key = resolve_paper_key(item)
        query = item["query"]
        spans = item["answer_spans"]
        chunks = chunks_by_paper.get(key, [])

        for mode in MODES:
            embed_spent = 0.0
            t0 = time.perf_counter()
            results = retrievers[mode.name](chunks, query)
            times[mode.name].append(time.perf_counter() - t0)
            if mode.name == "nugget":
                embed_times.append(embed_spent)

            hit = recall_at_k(results, spans, mode.field, normalize)
            flags[mode.name].append(hit)
            if hit:
                hits[mode.name] += 1
            else:
                misses[mode.name].append((i, query[:40], spans[0][:50]))
            mrr_sum[mode.name] += mrr_at_k(results, spans, mode.field, normalize)
            tokens[mode.name] += avg_tokens(results, mode.field, estimator)

    n = len(gold)
    if verbose:
        for mode in MODES:
            if mode.show_misses and misses[mode.name]:
                print(f"\n## {mode.label.capitalize()} Recall Misses:", file=sys.stderr)
                for idx, q, span in misses[mode.name][:5]:
                    print(f"  [{idx}] {q}... → {span}...", file=sys.stderr)

    latency: dict[str, dict] = {m.name: _latency_summary(times[m.name]) for m in MODES}
    if embed_fn is not None:
        latency["nugget_embed_calls"] = _latency_summary(embed_times)

    result: dict = {
        "match": match,
        "n_queries": n,
        "n_queries_without_chunks": n_queries_without_chunks,
        "modes": [m.name for m in MODES],
    }
    for m in MODES:
        result[m.name] = {
            "recall": round(hits[m.name] / n, 3) if n else 0,
            "mrr": round(mrr_sum[m.name] / n, 3) if n else 0,
            "avg_tokens": round(tokens[m.name] / n, 1) if n else 0,
        }
    result["latency_ms"] = latency
    # The paired comparison is defined between the reference (full_chunk) and nugget.
    result["comparison"] = compare(
        flags["full_chunk"],
        flags["nugget"],
        tokens["full_chunk"] / n if n else 0.0,
        tokens["nugget"] / n if n else 0.0,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=_existing_file)
    parser.add_argument("--gold", required=True, type=_existing_file)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--embedding-url",
        default=None,
        help="Embedding service URL (e.g. http://<internal-host>:9092). "
        "When set, nugget scoring uses BM25 + embedding hybrid.",
    )
    parser.add_argument(
        "--embed-weight",
        type=_float_between_0_1,
        default=0.5,
        help="Embedding weight in hybrid score: 0=BM25-only, 1=embed-only (default: 0.5)",
    )
    parser.add_argument(
        "--embedding-collection",
        default="search-engine",
        help="Embedding collection name (default: search-engine)",
    )
    parser.add_argument(
        "--token-estimator",
        choices=["words", "chars"],
        default="words",
        help="Token count estimator: words=split() (default), chars=len/4",
    )
    parser.add_argument(
        "--nugget-window",
        type=_non_negative_int,
        default=0,
        help="Also return N neighbouring sentences around each nugget (default: 0 = off)",
    )
    parser.add_argument(
        "--match",
        choices=["exact", "normalized"],
        default="exact",
        help="Span matching: exact = case-insensitive substring (default); normalized = also "
        "ignore whitespace/punctuation, for chunks from a different text extractor than the gold",
    )
    parser.add_argument(
        "--pool-label",
        default=None,
        help="Free-text note on how the chunk pool was built (e.g. 'api hybrid per-query, "
        "snippet_length=0, 2026-10-28'). Echoed in the JSON output as 'pool'.",
    )
    args = parser.parse_args()

    embed_fn = None
    if args.embedding_url:
        from nugget_rag.embedder import EmbedClient

        if not os.environ.get("EMBEDDING_API_KEY"):
            print("[WARN] EMBEDDING_API_KEY not set", file=sys.stderr)
        client = EmbedClient(
            args.embedding_url,
            collection=args.embedding_collection,
        )
        embed_fn = client.embed
        print(f"Embedding: {args.embedding_url}  weight={args.embed_weight}", file=sys.stderr)

    chunks_data: list[dict] = _load_json(args.chunks, "--chunks")
    gold: list[dict] = _load_json(args.gold, "--gold")

    try:
        validate_chunks(chunks_data)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    try:
        validate_gold(gold)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    chunks_by_paper = build_chunks_index(chunks_data)

    try:
        result = evaluate(
            chunks_by_paper,
            gold,
            top_k=args.top_k,
            verbose=args.verbose,
            embed_fn=embed_fn,
            embed_weight=args.embed_weight,
            estimator=args.token_estimator,
            nugget_window=args.nugget_window,
            match=args.match,
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    if args.pool_label:
        result["pool"] = args.pool_label

    tok_label = f"Avg tokens({args.token_estimator})"
    print(
        f"{'Mode':<18} {'Recall@' + str(args.top_k):<12} {'MRR@' + str(args.top_k):<12} {tok_label}"
    )
    print("-" * 60)
    for name in result["modes"]:
        m = result[name]
        label = mode_label(name, args.embed_weight if embed_fn else None)
        print(f"{label:<18} {m['recall']:<12} {m['mrr']:<12} {m['avg_tokens']}")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
