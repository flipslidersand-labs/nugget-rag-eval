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
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from nugget_rag.paper_registry import ARXIV_MAP as ARXIV_TO_PAPER_ID
from nugget_rag.paper_registry import PAPER_ID_TO_ARXIV

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _validate_url(url: str) -> None:
    """Reject non-HTTP(S) schemes to prevent SSRF via file://, ftp://, etc."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme!r}. Only 'http' and 'https' are allowed."
        )


# Unified timeout constants
_TIMEOUT_PAPERS = 15
_TIMEOUT_CHUNKS = 30


class FetchError(Exception):
    """Raised when an HTTP/network fetch fails."""


def fetch_papers(api_url: str, limit: int = 100) -> list[dict]:
    _validate_url(api_url)
    url = f"{api_url}/papers?{urlencode({'limit': limit, 'sort': 'score'})}"
    try:
        with urlopen(url, timeout=_TIMEOUT_PAPERS) as http_resp:
            return json.loads(http_resp.read())["papers"]
    except (URLError, HTTPError) as exc:
        raise FetchError(f"Failed to fetch papers: {exc}") from exc


def _sanitize_query(query: str) -> str:
    """Strip FTS5 special chars that cause SQLite parse errors (e.g. '-' → NOT operator)."""
    import re

    return re.sub(r"[^\w\s]", " ", query).strip()


def fetch_chunks_for_paper(api_url: str, paper_id: int, query: str, limit: int = 20) -> list[dict]:
    _validate_url(api_url)
    safe_query = _sanitize_query(query)
    url = f"{api_url}/search?{urlencode({'q': safe_query, 'mode': 'hybrid', 'paper_id': paper_id, 'limit': limit})}"
    try:
        with urlopen(url, timeout=_TIMEOUT_CHUNKS) as http_resp:
            results = json.loads(http_resp.read())["results"]
    except (URLError, HTTPError) as exc:
        raise FetchError(f"Failed to fetch chunks for paper {paper_id}: {exc}") from exc
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
    api_url: str,
    gold: list[dict],
    chunk_mode: str,
    target_tokens: int,
    fail_fast: bool = False,
) -> tuple[list[dict], int]:
    """Fetch chunks using each gold set query for its target paper.

    Returns a tuple of (deduplicated chunks, failure_count). A chunk is keyed
    by (paper_id, chunk_index) so the same physical chunk fetched by multiple
    queries is stored only once — the highest-scoring copy wins.

    Gold items may use ``arxiv_id`` (preferred) or ``paper_id`` (legacy).

    When *fail_fast* is ``False`` (default), individual fetch failures are
    logged as warnings and skipped so partial results are preserved.  When
    *fail_fast* is ``True``, the first ``FetchError`` is re-raised immediately.
    """
    if chunk_mode == "large":

        def key_field(c: dict) -> tuple:
            return (c["paper_id"], tuple(c["chunk_indices"]))

    else:

        def key_field(c: dict) -> tuple:
            return (c["paper_id"], c["chunk_index"])

    seen: dict[tuple, dict] = {}
    failure_count = 0
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
        try:
            raw = fetch_chunks_for_paper(api_url, pid, query, limit=30)
        except FetchError as exc:
            if fail_fast:
                raise
            print(f"  WARNING: {exc}, skipping paper {pid}", file=sys.stderr)
            failure_count += 1
            continue

        if chunk_mode == "large":
            chunks = combine_large_chunks(raw, pid, target_tokens)
        else:
            chunks = raw

        for c in chunks:
            k = key_field(c)
            if k not in seen or c["score"] > seen[k]["score"]:
                seen[k] = c

        print(
            f"  paper {pid} | query '{query[:40]}': {len(raw)} → {len(chunks)} chunks",
            file=sys.stderr,
        )

    return list(seen.values()), failure_count


def _positive_int(v: str) -> int:
    """Argparse type for integer arguments that must be >= 1."""
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


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
        type=_positive_int,
        default=512,
        help="Target token count for large chunk mode (must be >= 1)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Abort on first fetch error instead of skipping and continuing",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="Exit with code 1 when failure count exceeds this threshold (default: any failure exits 1)",
    )
    args = parser.parse_args()

    api = args.api_url.rstrip("/")
    _validate_url(api)
    failure_count = 0

    if args.gold_set:
        gold = json.loads(Path(args.gold_set).read_text(encoding="utf-8"))
        print(f"Per-query fetch for {len(gold)} gold items", file=sys.stderr)
        all_chunks, failure_count = fetch_per_query(
            api, gold, args.chunk_mode, args.large_chunk_target, fail_fast=args.fail_fast
        )
    else:
        try:
            papers = fetch_papers(api)
        except FetchError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(papers)} papers", file=sys.stderr)
        all_chunks = []
        for p in papers:
            pid = p["id"]
            try:
                raw = fetch_chunks_for_paper(api, pid, args.query)
            except FetchError as exc:
                if args.fail_fast:
                    print(f"[ERROR] {exc}", file=sys.stderr)
                    sys.exit(1)
                print(f"  WARNING: {exc}, skipping paper {pid}", file=sys.stderr)
                failure_count += 1
                continue
            if args.chunk_mode == "large":
                chunks = combine_large_chunks(raw, pid, args.large_chunk_target)
                print(f"  paper {pid}: {len(raw)} → {len(chunks)} large chunks", file=sys.stderr)
            else:
                chunks = raw
                print(f"  paper {pid}: {len(chunks)} chunks", file=sys.stderr)
            all_chunks.extend(chunks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(f"Wrote {len(all_chunks)} chunks to {out}", file=sys.stderr)

    if failure_count > 0:
        threshold = args.max_failures if args.max_failures is not None else 0
        if failure_count > threshold:
            print(
                f"[ERROR] {failure_count} fetch failure(s) exceeded threshold ({threshold})",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
