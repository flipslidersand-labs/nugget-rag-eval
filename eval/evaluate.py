"""Evaluate full-chunk vs nugget retrieval using Recall@k and token count.

Usage:
    python eval/evaluate.py --chunks data/chunks.json --gold eval/gold_set.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nugget_rag.retriever import retrieve_full_chunk, retrieve_nuggets


def recall_at_k(results: list[dict], answer_spans: list[str], field: str = "text") -> bool:
    haystack = " ".join(r.get(field, "") for r in results).lower()
    return any(span.lower() in haystack for span in answer_spans)


def avg_tokens(results: list[dict], field: str = "text") -> float:
    texts = [r.get(field, "") for r in results]
    return sum(len(t.split()) for t in texts) / max(len(texts), 1)


def evaluate(chunks_by_paper: dict[int, list[dict]], gold: list[dict], top_k: int = 5) -> dict:
    full_hits = nugget_hits = 0
    full_tokens = nugget_tokens = 0.0

    for item in gold:
        paper_id = item["paper_id"]
        query = item["query"]
        spans = item["answer_spans"]
        chunks = chunks_by_paper.get(paper_id, [])

        full = retrieve_full_chunk(chunks, query, top_k)
        nugget = retrieve_nuggets(chunks, query, top_k)

        if recall_at_k(full, spans, "text"):
            full_hits += 1
        if recall_at_k(nugget, spans, "nugget"):
            nugget_hits += 1

        full_tokens += avg_tokens(full, "text")
        nugget_tokens += avg_tokens(nugget, "nugget")

    n = len(gold)
    return {
        "n_queries": n,
        "full_chunk": {"recall": round(full_hits / n, 3) if n else 0, "avg_tokens": round(full_tokens / n, 1) if n else 0},
        "nugget": {"recall": round(nugget_hits / n, 3) if n else 0, "avg_tokens": round(nugget_tokens / n, 1) if n else 0},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks_data: list[dict] = json.loads(Path(args.chunks).read_text())
    gold: list[dict] = json.loads(Path(args.gold).read_text())

    chunks_by_paper: dict[int, list[dict]] = {}
    for c in chunks_data:
        chunks_by_paper.setdefault(c["paper_id"], []).append(c)

    result = evaluate(chunks_by_paper, gold, top_k=args.top_k)

    print(f"{'Mode':<14} {'Recall':<10} {'Avg tokens'}")
    print("-" * 36)
    print(f"{'full-chunk':<14} {result['full_chunk']['recall']:<10} {result['full_chunk']['avg_tokens']}")
    print(f"{'nugget':<14} {result['nugget']['recall']:<10} {result['nugget']['avg_tokens']}")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
