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
import sys
from collections.abc import Callable
from pathlib import Path

from nugget_rag.paper_registry import ARXIV_MAP
from nugget_rag.retriever import retrieve_full_chunk, retrieve_nuggets

_REQUIRED_FIELDS = frozenset({"query", "answer_spans"})


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


def _resolve_key(item: dict) -> str | int:
    """Resolve a gold item to its chunks_by_paper key (arxiv_id via ARXIV_MAP, else paper_id)."""
    arxiv_id = item.get("arxiv_id")
    if arxiv_id:
        return ARXIV_MAP.get(arxiv_id, arxiv_id)
    return item["paper_id"]


def check_gold_chunk_coverage(chunks_by_paper: dict[str, list[dict]], gold: list[dict]) -> int:
    """Warn about gold keys missing from chunks_by_paper; return the count of affected queries.

    Prints a ``[WARN]`` line to stderr with the count and up to 5 example keys.

    Raises:
        ValueError: If *gold* is non-empty and no gold key resolves to any chunks
            (every query would silently score recall 0).
    """
    missing = [key for item in gold if (key := _resolve_key(item)) not in chunks_by_paper]
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


def recall_at_k(results: list[dict], answer_spans: list[str], field: str = "text") -> bool:
    """Return True if any single result contains any answer span.

    Judged per-result (same criterion as :func:`mrr_at_k`) — never on joined
    text, which would create false positives when a span only matches across
    the boundary of two concatenated results (#146).
    """
    return any(
        any(span.lower() in text for span in answer_spans)
        for text in (r.get(field, "").lower() for r in results)
    )


def mrr_at_k(results: list[dict], answer_spans: list[str], field: str = "text") -> float:
    """Return reciprocal rank of the first hit (1-indexed), 0 if no hit."""
    for rank, result in enumerate(results, start=1):
        text = result.get(field, "").lower()
        if any(span.lower() in text for span in answer_spans):
            return 1.0 / rank
    return 0.0


def avg_tokens(results: list[dict], field: str = "text", estimator: str = "words") -> float:
    texts = [r.get(field, "") for r in results]
    if estimator == "chars":
        return sum(len(t) / 4 for t in texts) / max(len(texts), 1)
    return sum(len(t.split()) for t in texts) / max(len(texts), 1)


def evaluate(
    chunks_by_paper: dict[str, list[dict]],
    gold: list[dict],
    top_k: int = 5,
    verbose: bool = False,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    embed_weight: float = 0.5,
    estimator: str = "words",
) -> dict:
    validate_gold(gold)
    n_queries_without_chunks = check_gold_chunk_coverage(chunks_by_paper, gold)

    full_hits = nugget_hits = 0
    full_tokens = nugget_tokens = 0.0
    full_mrr_sum = nugget_mrr_sum = 0.0
    nugget_misses = []

    for i, item in enumerate(gold):
        # Support both arxiv_id (new) and paper_id (legacy)
        key = _resolve_key(item)
        query = item["query"]
        spans = item["answer_spans"]
        chunks = chunks_by_paper.get(key, [])

        full = retrieve_full_chunk(chunks, query, top_k)
        nugget = retrieve_nuggets(
            chunks,
            query,
            top_k,
            embed_fn=embed_fn,
            embed_weight=embed_weight,
        )

        full_hit = recall_at_k(full, spans, "text")
        nugget_hit = recall_at_k(nugget, spans, "nugget")

        if full_hit:
            full_hits += 1
        if nugget_hit:
            nugget_hits += 1
        else:
            nugget_misses.append((i, query[:40], spans[0][:50]))

        full_mrr_sum += mrr_at_k(full, spans, "text")
        nugget_mrr_sum += mrr_at_k(nugget, spans, "nugget")

        full_tokens += avg_tokens(full, "text", estimator)
        nugget_tokens += avg_tokens(nugget, "nugget", estimator)

    n = len(gold)
    if verbose and nugget_misses:
        print("\n## Nugget Recall Misses:", file=sys.stderr)
        for idx, q, span in nugget_misses[:5]:
            print(f"  [{idx}] {q}... → {span}...", file=sys.stderr)

    return {
        "n_queries": n,
        "n_queries_without_chunks": n_queries_without_chunks,
        "full_chunk": {
            "recall": round(full_hits / n, 3) if n else 0,
            "mrr": round(full_mrr_sum / n, 3) if n else 0,
            "avg_tokens": round(full_tokens / n, 1) if n else 0,
        },
        "nugget": {
            "recall": round(nugget_hits / n, 3) if n else 0,
            "mrr": round(nugget_mrr_sum / n, 3) if n else 0,
            "avg_tokens": round(nugget_tokens / n, 1) if n else 0,
        },
    }


def _float_between_0_1(v: str) -> float:
    """Argparse type for --embed-weight: must be in [0.0, 1.0]."""
    f = float(v)
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {f}")
    return f


def _positive_int(v: str) -> int:
    """Argparse type for --top-k: must be >= 1."""
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _existing_file(value: str) -> str:
    """argparse type= helper: abort early if the path does not point to a file."""
    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"file not found: {value}")
    return value


def _load_json(path: str, label: str) -> list[dict]:
    """Read *path* and parse JSON, exiting with a friendly message on failure."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[ERROR] {label} file not found: {path}")
    except PermissionError:
        sys.exit(f"[ERROR] permission denied reading {label} file: {path}")
    except UnicodeDecodeError:
        sys.exit(f"[ERROR] {label} file is not valid UTF-8: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"[ERROR] {label} file is not valid JSON ({path}): {exc}")


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
        validate_gold(gold)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    chunks_by_paper: dict[str | int, list[dict]] = {}
    for c in chunks_data:
        key = c.get("arxiv_id") or c["paper_id"]
        chunks_by_paper.setdefault(key, []).append(c)

    try:
        result = evaluate(
            chunks_by_paper,
            gold,
            top_k=args.top_k,
            verbose=args.verbose,
            embed_fn=embed_fn,
            embed_weight=args.embed_weight,
            estimator=args.token_estimator,
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    tok_label = f"Avg tokens({args.token_estimator})"
    mode_label = f"nugget(e{args.embed_weight:.1f})" if embed_fn else "nugget"
    print(
        f"{'Mode':<18} {'Recall@' + str(args.top_k):<12} {'MRR@' + str(args.top_k):<12} {tok_label}"
    )
    print("-" * 60)
    fc = result["full_chunk"]
    ng = result["nugget"]
    print(f"{'full-chunk':<18} {fc['recall']:<12} {fc['mrr']:<12} {fc['avg_tokens']}")
    print(f"{mode_label:<18} {ng['recall']:<12} {ng['mrr']:<12} {ng['avg_tokens']}")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
