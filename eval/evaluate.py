"""Evaluate full-chunk vs nugget retrieval using Recall@k and token count.

Usage:
    # BM25-only (default)
    python eval/evaluate.py --chunks data/chunks.json --gold eval/gold_set.json

    # BM25 + embedding hybrid nugget scoring
    python eval/evaluate.py --chunks data/chunks_large_perquery.json \
        --gold eval/gold_set.json \
        --embedding-url http://<internal-host>:9092 \
        --embedding-api-key <key> \
        --embed-weight 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from nugget_rag.retriever import retrieve_full_chunk, retrieve_nuggets


def recall_at_k(results: list[dict], answer_spans: list[str], field: str = "text") -> bool:
    haystack = " ".join(r.get(field, "") for r in results).lower()
    return any(span.lower() in haystack for span in answer_spans)


def avg_tokens(results: list[dict], field: str = "text") -> float:
    texts = [r.get(field, "") for r in results]
    return sum(len(t.split()) for t in texts) / max(len(texts), 1)


def evaluate(
    chunks_by_paper: dict[int, list[dict]],
    gold: list[dict],
    top_k: int = 5,
    verbose: bool = False,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    embed_weight: float = 0.5,
) -> dict:
    full_hits = nugget_hits = 0
    full_tokens = nugget_tokens = 0.0
    nugget_misses = []

    for i, item in enumerate(gold):
        paper_id = item["paper_id"]
        query = item["query"]
        spans = item["answer_spans"]
        chunks = chunks_by_paper.get(paper_id, [])

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

        full_tokens += avg_tokens(full, "text")
        nugget_tokens += avg_tokens(nugget, "nugget")

    n = len(gold)
    if verbose and nugget_misses:
        print("\n## Nugget Recall Misses:", file=sys.stderr)
        for idx, q, span in nugget_misses[:5]:
            print(f"  [{idx}] {q}... → {span}...", file=sys.stderr)

    return {
        "n_queries": n,
        "full_chunk": {
            "recall": round(full_hits / n, 3) if n else 0,
            "avg_tokens": round(full_tokens / n, 1) if n else 0,
        },
        "nugget": {
            "recall": round(nugget_hits / n, 3) if n else 0,
            "avg_tokens": round(nugget_tokens / n, 1) if n else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--embedding-url",
        default=None,
        help="Embedding service URL (e.g. http://<internal-host>:9092). "
        "When set, nugget scoring uses BM25 + embedding hybrid.",
    )
    parser.add_argument(
        "--embedding-api-key", default=None, help="X-API-Key for the embedding service"
    )
    parser.add_argument(
        "--embed-weight",
        type=float,
        default=0.5,
        help="Embedding weight in hybrid score: 0=BM25-only, 1=embed-only (default: 0.5)",
    )
    parser.add_argument(
        "--embedding-collection",
        default="search-engine",
        help="Embedding collection name (default: search-engine)",
    )
    args = parser.parse_args()

    embed_fn = None
    if args.embedding_url:
        from nugget_rag.embedder import EmbedClient

        client = EmbedClient(
            args.embedding_url,
            api_key=args.embedding_api_key or "",
            collection=args.embedding_collection,
        )
        embed_fn = client.embed
        print(f"Embedding: {args.embedding_url}  weight={args.embed_weight}", file=sys.stderr)

    chunks_data: list[dict] = json.loads(Path(args.chunks).read_text())
    gold: list[dict] = json.loads(Path(args.gold).read_text())

    chunks_by_paper: dict[int, list[dict]] = {}
    for c in chunks_data:
        chunks_by_paper.setdefault(c["paper_id"], []).append(c)

    result = evaluate(
        chunks_by_paper,
        gold,
        top_k=args.top_k,
        verbose=args.verbose,
        embed_fn=embed_fn,
        embed_weight=args.embed_weight,
    )

    mode_label = f"nugget(e{args.embed_weight:.1f})" if embed_fn else "nugget"
    print(f"{'Mode':<18} {'Recall':<10} {'Avg tokens'}")
    print("-" * 40)
    print(
        f"{'full-chunk':<18} {result['full_chunk']['recall']:<10} {result['full_chunk']['avg_tokens']}"
    )
    print(f"{mode_label:<18} {result['nugget']['recall']:<10} {result['nugget']['avg_tokens']}")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
