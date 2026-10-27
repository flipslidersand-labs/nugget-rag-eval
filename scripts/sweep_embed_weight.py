"""Sweep --embed-weight for nugget scoring, embedding every sentence only once (#199).

``eval/evaluate.py --embed-weight W`` re-embeds the same sentences for every W
although the vectors do not depend on W. This driver:

1. dry-runs ``evaluate()`` with a recorder to learn exactly which texts the real
   run will embed (no duplicated retrieval logic),
2. embeds the unique texts once, in batches, with progress and a resumable cache,
3. runs ``evaluate()`` for each weight against the cache (no network).

The API key is read from the EMBEDDING_API_KEY environment variable by
nugget_rag.embedder; it is never printed or written.

Usage:
    EMBEDDING_API_KEY=... python scripts/sweep_embed_weight.py \\
        --chunks pool.json --gold eval/gold_set.json eval/gold_set_hard.json \\
        --embedding-url http://<host>:9092 --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from eval.cli_helpers import existing_file, float_between_0_1, load_json, positive_int
from eval.evaluate import build_chunks_index, evaluate, validate_chunks, validate_gold

BATCH = 128  # texts per embedding request (the client allows up to 256)


def collect_texts(chunks_by_paper: dict, gold: list[dict], top_k: int, match: str) -> list[str]:
    """Texts evaluate() will send to embed_fn, found by a dry run with a recording stub."""
    seen: dict[str, None] = {}

    def recorder(texts: list[str]) -> list[list[float]]:
        seen.update(dict.fromkeys(texts))
        return [[0.0] for _ in texts]

    evaluate(chunks_by_paper, gold, top_k=top_k, embed_fn=recorder, match=match)
    return list(seen)


def load_cache(path: Path | None) -> dict[str, list[float]]:
    if path and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(path: Path | None, cache: dict[str, list[float]]) -> None:
    if path:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def warm_cache(
    texts: list[str],
    embed: Callable[[list[str]], list[list[float]]],
    cache: dict[str, list[float]],
    cache_path: Path | None = None,
    log=print,
) -> int:
    """Embed the texts missing from *cache*. Returns the number of texts newly embedded."""
    todo = [t for t in texts if t not in cache]
    log(f"embedding: {len(texts)} unique texts, {len(texts) - len(todo)} cached, {len(todo)} to do")
    t0 = time.perf_counter()
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        vecs = embed(batch)
        if len(vecs) != len(batch):
            raise RuntimeError(f"embedding service returned {len(vecs)} vectors for {len(batch)}")
        cache.update(zip(batch, vecs))
        save_cache(cache_path, cache)  # resumable if interrupted
        done = min(i + BATCH, len(todo))
        rate = done / max(time.perf_counter() - t0, 1e-9)
        eta = (len(todo) - done) / rate
        log(f"  {done}/{len(todo)}  {rate:.1f} texts/s  ETA {eta / 60:.1f} min")
    return len(todo)


def cached_embed_fn(cache: dict[str, list[float]]) -> Callable[[list[str]], list[list[float]]]:
    def embed(texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in cache]
        if missing:
            raise KeyError(
                f"{len(missing)} text(s) were not pre-embedded (first: {missing[0][:60]!r})"
            )
        return [cache[t] for t in texts]

    return embed


def sweep(
    chunks_by_paper: dict,
    golds: dict[str, list[dict]],
    weights: list[float],
    embed_fn: Callable[[list[str]], list[list[float]]],
    top_k: int = 5,
    match: str = "normalized",
) -> dict[str, dict]:
    out = {}
    for gname, gold in golds.items():
        for w in weights:
            out[f"{gname}_w{w}"] = evaluate(
                chunks_by_paper, gold, top_k=top_k, embed_fn=embed_fn, embed_weight=w, match=match
            )
    return out


def format_table(results: dict[str, dict]) -> str:
    lines = ["gold | w | full R/MRR | nugget R/MRR | nugget tok | 判別 p"]
    for key, d in results.items():
        g, w = key.rsplit("_w", 1)
        f, n, c = d["full_chunk"], d["nugget"], d["comparison"]
        lines.append(
            f"{g} | {w} | {f['recall']}/{f['mrr']} | {n['recall']}/{n['mrr']} | "
            f"{n['avg_tokens']} | {c['mcnemar_p']}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--chunks", required=True, type=existing_file)
    ap.add_argument("--gold", required=True, nargs="+", type=existing_file)
    ap.add_argument("--embedding-url", required=True)
    ap.add_argument(
        "--weights", nargs="+", type=float_between_0_1, default=[0.0, 0.3, 0.5, 0.7, 1.0]
    )
    ap.add_argument("--top-k", type=positive_int, default=5)
    ap.add_argument("--match", choices=["exact", "normalized"], default="normalized")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_json(args.chunks, "--chunks")
    try:
        validate_chunks(chunks)
        golds = {Path(p).stem: load_json(p, "--gold") for p in args.gold}
        for g in golds.values():
            validate_gold(g)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")
    by_paper = build_chunks_index(chunks)

    texts = list(
        dict.fromkeys(
            t
            for gold in golds.values()
            for t in collect_texts(by_paper, gold, args.top_k, args.match)
        )
    )
    cache_path = args.out_dir / "embed_cache.json"
    cache = load_cache(cache_path)

    from nugget_rag.embedder import EmbedClient

    client = EmbedClient(args.embedding_url)
    warm_cache(texts, client.embed, cache, cache_path)

    results = sweep(by_paper, golds, args.weights, cached_embed_fn(cache), args.top_k, args.match)
    (args.out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n" + format_table(results))
    print(f"\nresults: {args.out_dir / 'summary.json'}  (no API key inside)")


if __name__ == "__main__":
    main()
