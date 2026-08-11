"""Fetch paper chunks from academic-paper-system API for evaluation.

Usage:
    python scripts/fetch_papers.py --api-url http://localhost:8020 --out data/chunks.json
"""
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch_papers(api_url: str, limit: int = 100) -> list[dict]:
    url = f"{api_url}/papers?{urlencode({'limit': limit, 'sort': 'score'})}"
    return json.loads(urlopen(url, timeout=15).read())["papers"]


def fetch_chunks_for_paper(api_url: str, paper_id: int, query: str) -> list[dict]:
    url = f"{api_url}/search?{urlencode({'q': query, 'mode': 'hybrid', 'paper_id': paper_id, 'limit': 20})}"
    results = json.loads(urlopen(url, timeout=15).read())["results"]
    return [{"paper_id": paper_id, "chunk_index": r["chunk_index"], "text": r["snippet"], "score": r["score"]} for r in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8020")
    parser.add_argument("--out", default="data/chunks.json")
    parser.add_argument("--query", default="method results contribution")
    parser.add_argument("--chunk-mode", choices=["small", "large"], default="small",
                        help="small: individual chunks (~20 tokens), large: combined chunks (~512 tokens)")
    parser.add_argument("--large-chunk-target", type=int, default=512,
                        help="Target token count for large chunk mode")
    args = parser.parse_args()

    api = args.api_url.rstrip("/")
    papers = fetch_papers(api)
    print(f"Found {len(papers)} papers", file=sys.stderr)

    all_chunks = []
    for p in papers:
        pid = p["id"]
        chunks = fetch_chunks_for_paper(api, pid, args.query)

        if args.chunk_mode == "large":
            # Combine consecutive chunks until target size
            combined = []
            current = None
            current_size = 0
            for chunk in chunks:
                chunk_size = len(chunk["text"].split())
                if current is None:
                    current = {"paper_id": pid, "chunk_indices": [chunk["chunk_index"]],
                              "text": chunk["text"], "score": chunk["score"]}
                    current_size = chunk_size
                elif current_size + chunk_size <= args.large_chunk_target:
                    current["text"] += " " + chunk["text"]
                    current["chunk_indices"].append(chunk["chunk_index"])
                    current["score"] = max(current["score"], chunk["score"])
                    current_size += chunk_size
                else:
                    combined.append(current)
                    current = {"paper_id": pid, "chunk_indices": [chunk["chunk_index"]],
                              "text": chunk["text"], "score": chunk["score"]}
                    current_size = chunk_size
            if current:
                combined.append(current)
            all_chunks.extend(combined)
            print(f"  paper {pid}: {len(chunks)} → {len(combined)} large chunks", file=sys.stderr)
        else:
            all_chunks.extend(chunks)
            print(f"  paper {pid}: {len(chunks)} chunks", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2))
    print(f"Wrote {len(all_chunks)} chunks to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
