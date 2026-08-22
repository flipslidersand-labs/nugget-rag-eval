"""Recall@5 regression check for CI.

Runs BM25-only evaluation and fails if Recall@5 drops below the threshold.

Usage:
    python eval/check_regression.py \
        --chunks data/chunks_large.json \
        --gold eval/gold_set.json \
        --threshold 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@5 regression check")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON file")
    parser.add_argument("--gold", required=True, help="Path to gold set JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Minimum acceptable Recall@5 (default: 0.95)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval (default: 5)")
    args = parser.parse_args()

    chunks_data: list[dict] = json.loads(Path(args.chunks).read_text())
    gold: list[dict] = json.loads(Path(args.gold).read_text())

    chunks_by_paper: dict[str | int, list[dict]] = {}
    for c in chunks_data:
        key = c.get("arxiv_id") or c["paper_id"]
        chunks_by_paper.setdefault(key, []).append(c)

    result = evaluate(chunks_by_paper, gold, top_k=args.top_k)

    full_recall = result["full_chunk"]["recall"]
    nugget_recall = result["nugget"]["recall"]
    threshold = args.threshold

    print(f"Recall@{args.top_k} regression check")
    print(f"  threshold  : {threshold}")
    print(f"  full-chunk : {full_recall}")
    print(f"  nugget     : {nugget_recall}")

    failed = []
    if full_recall < threshold:
        failed.append(f"full-chunk Recall@{args.top_k} {full_recall} < threshold {threshold}")
    if nugget_recall < threshold:
        failed.append(f"nugget     Recall@{args.top_k} {nugget_recall} < threshold {threshold}")

    if failed:
        print("\n[FAIL] Recall regression detected:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)

    print(f"\n[PASS] Recall@{args.top_k} >= {threshold} for both modes.")


if __name__ == "__main__":
    main()
