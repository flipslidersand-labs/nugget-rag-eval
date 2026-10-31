"""Why does nugget miss? Classify each gold item and sweep nuggets_per_chunk x window (#201).

For every gold item (searching only the gold paper's chunks, like evaluate.py):

  A retrieval miss   no top-k chunk contains the answer span (full-chunk misses too)
  B split miss       a top-k chunk contains the span but no single sentence does
  C extraction miss  a sentence has the span but is not among the nugget sentences
  D hit              the nugget text contains the span

The sweep re-runs retrieve_nuggets with several nuggets_per_chunk / window values.
Selection logic comes from nugget_rag itself, nothing is re-implemented here.

Usage:
    python scripts/analyze_nugget_failures.py --chunks pool.json --gold eval/gold_set_hard.json \\
        --match normalized --sweep
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from eval.cli_helpers import existing_file, load_json, positive_int
from eval.evaluate import (
    _span_in,
    avg_tokens,
    build_chunks_index,
    recall_at_k,
    validate_chunks,
    validate_gold,
)
from nugget_rag.chunker import split_sentences
from nugget_rag.paper_registry import resolve_paper_key
from nugget_rag.retriever import retrieve_full_chunk, retrieve_nuggets
from nugget_rag.scorer import bm25_scores, top_nuggets

LABELS = {
    "A": "A 検索の失敗（根拠チャンクが上位に入らない）",
    "B": "B 文分割の失敗（根拠が文をまたぐ）",
    "C": "C 抽出の失敗（チャンクは入ったが根拠文が選ばれない）",
    "D": "D 成功",
}


def classify(
    chunks: list[dict], query: str, spans: list[str], top_k: int, per_chunk: int, normalize: bool
) -> tuple[str, int | None]:
    """Return (class, rank of the best sentence holding the span among its chunk's sentences)."""
    top = retrieve_full_chunk(chunks, query, top_k)
    holders = [c for c in top if _span_in(c.get("text", ""), spans, normalize)]
    if not holders:
        return "A", None
    sentences = split_sentences(holders[0]["text"])
    holding = [i for i, s in enumerate(sentences) if _span_in(s, spans, normalize)]
    if not holding:
        return "B", None
    chosen = top_nuggets(query, sentences, top_k=per_chunk)
    if any(_span_in(s, spans, normalize) for s in chosen):
        return "D", None
    scores = bm25_scores(query, sentences)
    order = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    return "C", min(order.index(i) for i in holding) + 1


def classify_all(
    by_paper: dict, gold: list[dict], top_k: int = 5, per_chunk: int = 3, normalize: bool = False
) -> tuple[Counter, Counter]:
    """Counts per class, and for class C a count per rank bucket (<=5, <=8, >8)."""
    classes: Counter = Counter()
    c_rank: Counter = Counter()
    for item in gold:
        chunks = by_paper.get(resolve_paper_key(item), [])
        cls, rank = classify(
            chunks, item["query"], item["answer_spans"], top_k, per_chunk, normalize
        )
        classes[cls] += 1
        if cls == "C":
            c_rank["<=5" if rank <= 5 else "<=8" if rank <= 8 else ">8"] += 1
    return classes, c_rank


def sweep(
    by_paper: dict,
    gold: list[dict],
    ks: tuple[int, ...] = (1, 3, 5, 8),
    windows: tuple[int, ...] = (0, 1),
    top_k: int = 5,
    normalize: bool = False,
) -> list[dict]:
    rows = []
    for k in ks:
        for w in windows:
            hits, toks = 0, 0.0
            for item in gold:
                chunks = by_paper.get(resolve_paper_key(item), [])
                res = retrieve_nuggets(chunks, item["query"], top_k, nuggets_per_chunk=k, window=w)
                hits += recall_at_k(res, item["answer_spans"], "nugget", normalize)
                toks += avg_tokens(res, "nugget")
            n = max(len(gold), 1)
            rows.append(
                {
                    "k": k,
                    "window": w,
                    "recall": round(hits / n, 3),
                    "avg_tokens": round(toks / n, 1),
                }
            )
    return rows


def format_report(classes: Counter, c_rank: Counter, n: int) -> str:
    lines = [f"n={n}"]
    for key in "ABCD":
        lines.append(f"  {LABELS[key]}: {classes.get(key, 0)}")
        if key == "C" and classes.get("C"):
            lines.append(
                "    根拠文の順位: "
                + ", ".join(f"{b} {c_rank.get(b, 0)}" for b in ("<=5", "<=8", ">8"))
            )
    return "\n".join(lines)


def format_sweep(rows: list[dict]) -> str:
    out = ["k  window  Recall  avg_tokens/chunk"]
    out += [f"{r['k']:<3}{r['window']:<8}{r['recall']:<8}{r['avg_tokens']}" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunks", required=True, type=existing_file)
    ap.add_argument("--gold", required=True, type=existing_file)
    ap.add_argument("--top-k", type=positive_int, default=5)
    ap.add_argument("--nuggets-per-chunk", type=positive_int, default=3)
    ap.add_argument("--match", choices=["exact", "normalized"], default="exact")
    ap.add_argument("--sweep", action="store_true", help="also sweep nuggets_per_chunk x window")
    args = ap.parse_args()

    chunks, gold = load_json(args.chunks, "--chunks"), load_json(args.gold, "--gold")
    try:
        validate_chunks(chunks)
        validate_gold(gold)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")
    by_paper, normalize = build_chunks_index(chunks), args.match == "normalized"

    classes, c_rank = classify_all(by_paper, gold, args.top_k, args.nuggets_per_chunk, normalize)
    print(format_report(classes, c_rank, len(gold)))
    if args.sweep:
        print("\n" + format_sweep(sweep(by_paper, gold, top_k=args.top_k, normalize=normalize)))


if __name__ == "__main__":
    main()
