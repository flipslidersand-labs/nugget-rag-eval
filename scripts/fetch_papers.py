"""Fetch paper chunks from academic-paper-system API for evaluation.

Usage:
    # Generic query (baseline)
    python scripts/fetch_papers.py --api-url http://localhost:8020 --out data/chunks.json

    # Per-evaluation-query fetch (recommended for accurate Recall measurement)
    python scripts/fetch_papers.py --api-url http://localhost:8020 --out data/chunks.json \
        --gold-set eval/gold_set.json
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from nugget_rag.paper_registry import ARXIV_MAP as ARXIV_TO_PAPER_ID
from nugget_rag.paper_registry import PAPER_ID_TO_ARXIV


def fetch_papers(api_url: str, limit: int = 100) -> list[dict]:
    url = f"{api_url}/papers?{urlencode({'limit': limit, 'sort': 'score'})}"
    try:
        return json.loads(urlopen(url, timeout=15).read())["papers"]
    except (URLError, HTTPError) as exc:
        print(f"[ERROR] Failed to fetch papers: {exc}", file=sys.stderr)
        sys.exit(1)


def _sanitize_query(query: str) -> str:
    """Strip FTS5 special chars that cause SQLite parse errors (e.g. '-' → NOT operator)."""
    import re

    return re.sub(r"[^\w\s]", " ", query).strip()


def fetch_chunks_for_paper(api_url: str, paper_id: int, query: str, limit: int = 20) -> list[dict]:
    safe_query = _sanitize_query(query)
    url = f"{api_url}/search?{urlencode({'q': safe_query, 'mode': 'hybrid', 'paper_id': paper_id, 'limit': limit})}"
    try:
        results = json.loads(urlopen(url, timeout=30).read())["results"]
    except (URLError, HTTPError) as exc:
        print(f"[ERROR] Failed to fetch chunks for paper {paper_id}: {exc}", file=sys.stderr)
        sys.exit(1)
    arxiv_id = PAPER_ID_TO_ARXIV.get(paper_id)
    chunks = []
    for r in results:
        chunk = {
            "paper_id": paper_id,
            "chunk_index": r["chunk_index"],
            "text": r["snippet"],
            "score": r["score"],
        }
        if arxiv_id is not None:
            chunk["arxiv_id"] = arxiv_id
        chunks.append(chunk)
    return chunks


def combine_large_chunks(chunks: list[dict], paper_id: int, target_tokens: int) -> list[dict]:
    """Merge consecutive chunks up to target_tokens words."""
    arxiv_id = PAPER_ID_TO_ARXIV.get(paper_id)
    combined = []
    current = None
    current_size = 0
    for chunk in chunks:
        chunk_size = len(chunk["text"].split())
        if current is None:
            current = {
                "paper_id": paper_id,
                "chunk_indices": [chunk["chunk_index"]],
                "text": chunk["text"],
                "score": chunk["score"],
            }
            if arxiv_id is not None:
                current["arxiv_id"] = arxiv_id
            current_size = chunk_size
        elif current_size + chunk_size <= target_tokens:
            current["text"] += " " + chunk["text"]
            current["chunk_indices"].append(chunk["chunk_index"])
            current["score"] = max(current["score"], chunk["score"])
            current_size += chunk_size
        else:
            combined.append(current)
            current = {
                "paper_id": paper_id,
                "chunk_indices": [chunk["chunk_index"]],
                "text": chunk["text"],
                "score": chunk["score"],
            }
            if arxiv_id is not None:
                current["arxiv_id"] = arxiv_id
            current_size = chunk_size
    if current:
        combined.append(current)
    return combined


def fetch_per_query(
    api_url: str, gold: list[dict], chunk_mode: str, target_tokens: int
) -> list[dict]:
    """Fetch chunks using each gold set query for its target paper.

    Returns deduplicated chunks. A chunk is keyed by (paper_id, chunk_index) so
    the same physical chunk fetched by multiple queries is stored only once —
    the highest-scoring copy wins.

    Gold items may use ``arxiv_id`` (preferred) or ``paper_id`` (legacy).
    """
    seen: dict[tuple, dict] = {}
    for item in gold:
        # Resolve paper_id: prefer arxiv_id field, fall back to paper_id int
        if "arxiv_id" in item:
            pid = ARXIV_TO_PAPER_ID.get(item["arxiv_id"])
            if pid is None:
                print(
                    f"  WARNING: unknown arxiv_id '{item['arxiv_id']}', skipping", file=sys.stderr
                )
                continue
        else:
            pid = item["paper_id"]
        query = item["query"]
        raw = fetch_chunks_for_paper(api_url, pid, query, limit=30)

        if chunk_mode == "large":
            chunks = combine_large_chunks(raw, pid, target_tokens)

            def key_field(c):
                return (c["paper_id"], tuple(c["chunk_indices"]))
        else:
            chunks = raw

            def key_field(c):
                return (c["paper_id"], c["chunk_index"])

        for c in chunks:
            k = key_field(c)
            if k not in seen or c["score"] > seen[k]["score"]:
                seen[k] = c

        print(
            f"  paper {pid} | query '{query[:40]}': {len(raw)} → {len(chunks)} chunks",
            file=sys.stderr,
        )

    return list(seen.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8020")
    parser.add_argument("--out", default="data/chunks.json")
    parser.add_argument(
        "--query",
        default="method results contribution",
        help="Generic query used when --gold-set is not provided",
    )
    parser.add_argument(
        "--gold-set",
        default=None,
        help="Path to gold_set.json. When provided, fetches chunks per evaluation query "
        "for each paper so relevant chunks are guaranteed to be in the pool.",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=["small", "large"],
        default="small",
        help="small: individual chunks (~20 tokens), large: combined chunks (~512 tokens)",
    )
    parser.add_argument(
        "--large-chunk-target",
        type=int,
        default=512,
        help="Target token count for large chunk mode",
    )
    args = parser.parse_args()

    api = args.api_url.rstrip("/")

    if args.gold_set:
        gold = json.loads(Path(args.gold_set).read_text())
        print(f"Per-query fetch for {len(gold)} gold items", file=sys.stderr)
        all_chunks = fetch_per_query(api, gold, args.chunk_mode, args.large_chunk_target)
    else:
        papers = fetch_papers(api)
        print(f"Found {len(papers)} papers", file=sys.stderr)
        all_chunks = []
        for p in papers:
            pid = p["id"]
            raw = fetch_chunks_for_paper(api, pid, args.query)
            if args.chunk_mode == "large":
                chunks = combine_large_chunks(raw, pid, args.large_chunk_target)
                print(f"  paper {pid}: {len(raw)} → {len(chunks)} large chunks", file=sys.stderr)
            else:
                chunks = raw
                print(f"  paper {pid}: {len(chunks)} chunks", file=sys.stderr)
            all_chunks.extend(chunks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2))
    print(f"Wrote {len(all_chunks)} chunks to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
