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

from eval.cli_helpers import (
    add_embedding_args,
    build_embed_client,
    existing_file,
    float_between_0_1,
    positive_int,
    write_json_atomic,
)
from eval.evaluate import (
    evaluate,
    format_diagnostics,
    load_chunks_index,
    load_gold,
    pop_diagnostics,
)
from nugget_rag.embedder import validate_vectors

BATCH = 128  # texts per embedding request (the client allows up to 256)


def collect_texts(chunks_by_paper: dict, gold: list[dict], top_k: int, match: str) -> list[str]:
    """Texts evaluate() will send to embed_fn, found by a dry run with a recording stub."""
    seen: dict[str, None] = {}

    def recorder(texts: list[str]) -> list[list[float]]:
        seen.update(dict.fromkeys(texts))
        return [[0.0] for _ in texts]

    # weight must be non-zero: 0 skips embed_fn (#239); the texts do not depend on its value
    evaluate(chunks_by_paper, gold, top_k=top_k, embed_fn=recorder, embed_weight=1.0, match=match)
    return list(seen)


def load_cache(path: Path | None) -> dict[str, list[float]]:
    if not (path and path.is_file()):
        return {}
    # Re-embedding is expensive, so a corrupt cache is an error, never silently discarded.
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except PermissionError:
        sys.exit(f"[ERROR] permission denied reading cache file: {path}")
    except OSError as exc:
        sys.exit(f"[ERROR] cannot read cache file: {path} ({exc})")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        sys.exit(f"[ERROR] cache file is corrupt: {path} ({exc})")
    if not isinstance(cache, dict):
        sys.exit(f"[ERROR] cache file is corrupt: {path} (expected a JSON object)")
    for key, vec in cache.items():
        if not (
            isinstance(vec, list)
            and vec
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vec)
        ):
            sys.exit(f"[ERROR] cache file is corrupt: {path} (entry {key[:60]!r} is not a vector)")
    return cache


def save_cache(path: Path | None, cache: dict[str, list[float]]) -> None:
    if path:
        write_json_atomic(path, cache, indent=None)


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
        validate_vectors(vecs)  # never cache a vector load_cache() would reject (#326)
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
) -> dict[tuple[str, float], dict]:
    out = {}
    for gname, gold in golds.items():
        for i, w in enumerate(weights):
            out[(gname, w)] = evaluate(
                chunks_by_paper, gold, top_k=top_k, embed_fn=embed_fn, embed_weight=w, match=match
            )
            if i == 0:  # coverage does not depend on the weight: show it once per gold
                for line in format_diagnostics(out[(gname, w)], prefix=f"[gold={gname}] "):
                    print(line, file=sys.stderr)
            pop_diagnostics(out[(gname, w)])
    return out


def format_table(results: dict[tuple[str, float], dict]) -> str:
    lines = ["gold | w | full R/MRR | nugget R/MRR | nugget tok | 判別 p"]
    for (g, w), d in results.items():
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
    add_embedding_args(ap, weight=False, url_required=True)
    ap.add_argument(
        "--weights", nargs="+", type=float_between_0_1, default=[0.0, 0.3, 0.5, 0.7, 1.0]
    )
    ap.add_argument("--top-k", type=positive_int, default=5)
    ap.add_argument("--match", choices=["exact", "normalized"], default="normalized")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    by_paper = load_chunks_index(args.chunks, "--chunks")
    golds = {}
    for p in args.gold:
        gname = Path(p).stem
        if gname in golds:
            sys.exit(f"[ERROR] gold name '{gname}' is used twice ({p})")
        golds[gname] = load_gold(p, "--gold", error_prefix=f"gold={gname}: ")

    texts: dict[str, None] = {}
    for gname, gold in golds.items():
        try:
            texts.update(dict.fromkeys(collect_texts(by_paper, gold, args.top_k, args.match)))
        except ValueError as exc:
            sys.exit(f"[ERROR] gold={gname}: {exc}")

    from nugget_rag.embedder import EmbedError

    cache_path = args.out_dir / "embed_cache.json"
    cache = load_cache(cache_path)
    client = build_embed_client(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        warm_cache(list(texts), client.embed, cache, cache_path)
    except EmbedError as exc:
        sys.exit(f"[ERROR] embedding service failed: {exc}")
    except RuntimeError as exc:  # warm_cache: vector count != batch size
        sys.exit(f"[ERROR] {exc}")

    # collect_texts embeds with dummy 1-dim vectors, so a real dimension mismatch
    # (mixed-model cache, inconsistent service) only surfaces in sweep().
    try:
        results = sweep(
            by_paper, golds, args.weights, cached_embed_fn(cache), args.top_k, args.match
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")
    summary = {f"{g}_w{w}": d for (g, w), d in results.items()}  # on-disk key format unchanged
    write_json_atomic(args.out_dir / "summary.json", summary, indent=1)
    print("\n" + format_table(results))
    print(f"\nresults: {args.out_dir / 'summary.json'}  (no API key inside)")


if __name__ == "__main__":
    main()
