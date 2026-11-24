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
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from eval.cli_helpers import add_embedding_args, build_embed_client
from eval.cli_helpers import existing_file as _existing_file
from eval.cli_helpers import load_json as _load_json
from eval.cli_helpers import non_negative_int as _non_negative_int
from eval.cli_helpers import positive_int as _positive_int
from eval.compare import compare
from eval.metrics import avg_tokens, check_estimator, mrr_at_k, percentile, recall_at_k
from eval.schema import (
    build_chunks_index,
    check_gold_chunk_coverage,
    validate_chunks,
    validate_gold,
)
from nugget_rag.embedder import EmbedError
from nugget_rag.paper_registry import resolve_paper_key
from nugget_rag.retriever import rank_chunks, retrieve_full_chunk, retrieve_nuggets


def _validated_or_exit(check: Callable[[list[dict]], None], data: list[dict], prefix: str) -> None:
    """Run *check* on *data*; a ValueError becomes ``sys.exit("[ERROR] <prefix><message>")``."""
    try:
        check(data)
    except ValueError as exc:
        sys.exit(f"[ERROR] {prefix}{exc}")


def load_chunks_index(
    path: str, label: str, *, error_prefix: str = ""
) -> dict[str | int, list[dict]]:
    """Read + validate a chunks JSON and return it indexed by paper (#340).

    The one place the CLIs turn a bad file into ``sys.exit("[ERROR] ...")``. *label* names the
    option in read errors (``--chunks``); *error_prefix* is put before a validation message
    (e.g. ``"pool=P: "``).
    """
    chunks = _load_json(path, label)
    _validated_or_exit(validate_chunks, chunks, error_prefix)
    return build_chunks_index(chunks)


def load_gold(path: str, label: str, *, error_prefix: str = "") -> list[dict]:
    """Read + validate a gold JSON; same exit behaviour as :func:`load_chunks_index` (#340)."""
    gold = _load_json(path, label)
    _validated_or_exit(validate_gold, gold, error_prefix)
    return gold


def load_inputs(chunks_path: str, gold_path: str) -> tuple[dict[str | int, list[dict]], list[dict]]:
    """Load ``--chunks`` and ``--gold`` the way evaluate/check_regression/analyze all need (#340).

    Both files are read first, then chunks and gold are validated (in that order), so the
    first error reported is the same as before the CLIs shared this. Gold coverage
    (``check_gold_chunk_coverage``) is deliberately not done here: ``evaluate()`` does it itself.
    """
    chunks = _load_json(chunks_path, "--chunks")
    gold = _load_json(gold_path, "--gold")
    _validated_or_exit(validate_chunks, chunks, "")
    _validated_or_exit(validate_gold, gold, "")
    return build_chunks_index(chunks), gold


def _latency_summary(values_s: list[float]) -> dict:
    return {
        "p50": round(percentile(values_s, 50) * 1000, 2),
        "p95": round(percentile(values_s, 95) * 1000, 2),
    }


EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class RetrievalMode:
    """One retrieval mode to evaluate. Add an entry to MODES to add a mode."""

    name: str  # key in the result dict (and in baseline files)
    field: str  # chunk field holding the text that is judged and counted
    label: str  # display name
    # (chunks of one paper, query, config, embed_fn) -> top-k results; with uses_ranking a fifth
    # argument is the rank_chunks() result for (chunks, query), computed once per query (#390)
    retrieve: Callable[..., list[dict]]
    show_misses: bool = False  # list recall misses under --verbose
    uses_embedding: bool = False  # calls embed_fn: its time is reported, the label shows the weight
    uses_ranking: bool = False  # retrieve takes the shared BM25 chunk ranking as a fifth argument


def _retrieve_full_chunk(chunks, query, cfg, embed_fn, ranked):
    return retrieve_full_chunk(chunks, query, cfg.top_k, ranked=ranked)


def _retrieve_nugget(chunks, query, cfg, embed_fn, ranked):
    return retrieve_nuggets(
        chunks,
        query,
        cfg.top_k,
        nuggets_per_chunk=cfg.nuggets_per_chunk,
        embed_fn=embed_fn,
        embed_weight=cfg.embed_weight,
        window=cfg.nugget_window,
        ranked=ranked,
    )


MODES: tuple[RetrievalMode, ...] = (
    RetrievalMode("full_chunk", "text", "full-chunk", _retrieve_full_chunk, uses_ranking=True),
    RetrievalMode(
        "nugget",
        "nugget",
        "nugget",
        _retrieve_nugget,
        show_misses=True,
        uses_embedding=True,
        uses_ranking=True,
    ),
)


def mode_label(name: str, embed_weight: float | None = None) -> str:
    """Display label for a mode; an embedding mode shows the embed weight when embeddings are used."""
    mode = next(m for m in MODES if m.name == name)
    if mode.uses_embedding and embed_weight is not None:
        return f"{mode.label}(e{embed_weight:.1f})"
    return mode.label


@dataclass(frozen=True)
class EvalConfig:
    """Retrieval settings of one evaluate() run (#392). Defaults are the CLI defaults.

    ``embed_fn`` is not here: it is a live client, not a setting. Invalid ``match`` /
    ``estimator`` raise ValueError at construction.
    """

    top_k: int = 5
    match: str = "exact"
    estimator: str = "words"
    nugget_window: int = 0
    nuggets_per_chunk: int = 3
    embed_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.match not in ("exact", "normalized"):
            raise ValueError(f"match must be 'exact' or 'normalized', got {self.match!r}")
        check_estimator(self.estimator)


def format_diagnostics(result: dict, verbose: bool = False, prefix: str = "") -> list[str]:
    """Stderr lines for ``result["warnings"]`` and, if *verbose*, the first 5 recall misses.

    *prefix* is put before each warning (e.g. ``"[pool=A gold=B] "``) so a driver that
    evaluates many pool/gold pairs can say which one a warning belongs to.
    """
    lines = [prefix + w for w in result.get("warnings", [])]
    if verbose:
        for mode in MODES:
            misses = result.get("misses", {}).get(mode.name, [])
            if mode.show_misses and misses:
                lines.append(f"\n## {mode.label.capitalize()} Recall Misses:")
                lines += [f"  [{m['index']}] {m['query']}... → {m['span']}..." for m in misses[:5]]
    return lines


def pop_diagnostics(result: dict) -> dict:
    """Remove the stderr-only keys (``warnings``, ``misses``) from *result* and return it.

    Keeps JSON the tools write out (stdout, --json, summary.json) free of the extra keys.
    """
    result.pop("warnings", None)
    result.pop("misses", None)
    return result


def _fold_legacy(config: EvalConfig | None, legacy: dict) -> EvalConfig:
    """Merge the old per-setting keyword arguments of evaluate() into one EvalConfig.

    Only the legacy arguments the caller actually passed (not None) count. Passing one together
    with *config* is ambiguous, so it is a ValueError rather than one silently winning.
    """
    given = {k: v for k, v in legacy.items() if v is not None}
    if config is None:
        return EvalConfig(**given)
    if given:
        raise ValueError(
            f"pass either config or the legacy keyword arguments {sorted(given)}, not both"
        )
    return config


def evaluate(
    chunks_by_paper: dict[str, list[dict]],
    gold: list[dict],
    config: EvalConfig | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    *,
    top_k: int | None = None,
    embed_weight: float | None = None,
    estimator: str | None = None,
    nugget_window: int | None = None,
    match: str | None = None,
) -> dict:
    """Run every retrieval mode over *gold* and return recall / MRR / tokens per mode.

    Settings come from *config* (default ``EvalConfig()``). The keyword arguments ``top_k``,
    ``embed_weight``, ``estimator``, ``nugget_window`` and ``match`` are the old way to set the
    same fields; they fold into a config, and mixing them with *config* raises ValueError.
    """
    cfg = _fold_legacy(
        config,
        {
            "top_k": top_k,
            "embed_weight": embed_weight,
            "estimator": estimator,
            "nugget_window": nugget_window,
            "match": match,
        },
    )
    top_k, match, estimator = cfg.top_k, cfg.match, cfg.estimator
    normalize = match == "normalized"
    validate_gold(gold)
    n_queries_without_chunks, warnings = check_gold_chunk_coverage(chunks_by_paper, gold)

    embed_spent = 0.0
    timed_embed_fn = embed_fn
    if embed_fn is not None:

        def timed_embed_fn(texts):  # noqa: F811 - wraps embed_fn to time HTTP calls
            nonlocal embed_spent
            t0 = time.perf_counter()
            try:
                return embed_fn(texts)
            finally:
                embed_spent += time.perf_counter() - t0

    hits = {m.name: 0 for m in MODES}
    tokens = {m.name: 0.0 for m in MODES}
    mrr_sum = {m.name: 0.0 for m in MODES}
    times: dict[str, list[float]] = {m.name: [] for m in MODES}
    flags: dict[str, list[bool]] = {m.name: [] for m in MODES}
    misses: dict[str, list[dict]] = {m.name: [] for m in MODES}
    embed_times: list[float] = []

    for i, item in enumerate(gold):
        # Support both arxiv_id (new) and paper_id (legacy)
        key = resolve_paper_key(item)
        query = item["query"]
        spans = item["answer_spans"]
        chunks = chunks_by_paper.get(key, [])

        # One BM25 ranking per query, shared by the modes that use it. Its time is added to each
        # of them, so every mode's latency is what it costs on its own (#390).
        ranked, rank_time = None, 0.0
        if any(m.uses_ranking for m in MODES):
            t0 = time.perf_counter()
            ranked = rank_chunks(chunks, query)
            rank_time = time.perf_counter() - t0

        for mode in MODES:
            embed_spent = 0.0
            t0 = time.perf_counter()
            if mode.uses_ranking:
                results = mode.retrieve(chunks, query, cfg, timed_embed_fn, ranked)
            else:
                results = mode.retrieve(chunks, query, cfg, timed_embed_fn)
            times[mode.name].append(
                time.perf_counter() - t0 + (rank_time if mode.uses_ranking else 0.0)
            )
            if mode.uses_embedding:
                embed_times.append(embed_spent)

            hit = recall_at_k(results, spans, mode.field, normalize)
            flags[mode.name].append(hit)
            if hit:
                hits[mode.name] += 1
            else:
                misses[mode.name].append({"index": i, "query": query[:40], "span": spans[0][:50]})
            mrr_sum[mode.name] += mrr_at_k(results, spans, mode.field, normalize)
            tokens[mode.name] += avg_tokens(results, mode.field, estimator)

    n = len(gold)
    latency: dict[str, dict] = {m.name: _latency_summary(times[m.name]) for m in MODES}
    if embed_fn is not None:
        latency["nugget_embed_calls"] = _latency_summary(embed_times)

    result: dict = {
        "match": match,
        "n_queries": n,
        "n_queries_without_chunks": n_queries_without_chunks,
        # Diagnostics for the caller to show (evaluate() prints nothing); see format_diagnostics.
        "warnings": warnings,
        "misses": misses,
        "modes": [m.name for m in MODES],
        # The settings of this run, so a baseline can be written from / checked against it (#394).
        "config": {
            "top_k": cfg.top_k,
            "match": cfg.match,
            "nugget_window": cfg.nugget_window,
            "embed_weight": cfg.embed_weight,
            "nuggets_per_chunk": cfg.nuggets_per_chunk,
        },
        # Unrounded hit counts per mode: gates compare hits / n_queries, not the rounded recall.
        "hits": {m.name: hits[m.name] for m in MODES},
    }
    for m in MODES:
        result[m.name] = {
            "recall": round(hits[m.name] / n, 3) if n else 0,
            "mrr": round(mrr_sum[m.name] / n, 3) if n else 0,
            "avg_tokens": round(tokens[m.name] / n, 1) if n else 0,
        }
    result["latency_ms"] = latency
    # The paired comparison is defined between the reference (full_chunk) and nugget.
    result["comparison"] = compare(
        flags["full_chunk"],
        flags["nugget"],
        tokens["full_chunk"] / n if n else 0.0,
        tokens["nugget"] / n if n else 0.0,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=_existing_file)
    parser.add_argument("--gold", required=True, type=_existing_file)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--verbose", action="store_true")
    add_embedding_args(parser)
    parser.add_argument(
        "--token-estimator",
        choices=["words", "chars"],
        default="words",
        help="Token count estimator: words=split() (default), chars=len/4",
    )
    parser.add_argument(
        "--nugget-window",
        type=_non_negative_int,
        default=0,
        help="Also return N neighbouring sentences around each nugget (default: 0 = off)",
    )
    parser.add_argument(
        "--nuggets-per-chunk",
        type=_positive_int,
        default=3,
        help="Sentences selected from each top-k chunk for the nugget mode (default: 3)",
    )
    parser.add_argument(
        "--match",
        choices=["exact", "normalized"],
        default="exact",
        help="Span matching: exact = case-insensitive substring (default); normalized = also "
        "ignore whitespace/punctuation, for chunks from a different text extractor than the gold",
    )
    parser.add_argument(
        "--pool-label",
        default=None,
        help="Free-text note on how the chunk pool was built (e.g. 'api hybrid per-query, "
        "snippet_length=0, 2026-10-28'). Echoed in the JSON output as 'pool'.",
    )
    args = parser.parse_args()

    embed_fn = None
    if args.embedding_url:
        embed_fn = build_embed_client(args).embed
        print(f"Embedding: {args.embedding_url}  weight={args.embed_weight}", file=sys.stderr)

    chunks_by_paper, gold = load_inputs(args.chunks, args.gold)

    try:
        result = evaluate(
            chunks_by_paper,
            gold,
            EvalConfig(
                top_k=args.top_k,
                match=args.match,
                estimator=args.token_estimator,
                nugget_window=args.nugget_window,
                nuggets_per_chunk=args.nuggets_per_chunk,
                embed_weight=args.embed_weight,
            ),
            embed_fn=embed_fn,
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")
    except EmbedError as exc:
        sys.exit(f"[ERROR] embedding service failed: {exc}")

    for line in format_diagnostics(result, verbose=args.verbose):
        print(line, file=sys.stderr)
    pop_diagnostics(result)

    if args.pool_label:
        result["pool"] = args.pool_label

    tok_label = f"Avg tokens({args.token_estimator})"
    print(
        f"{'Mode':<18} {'Recall@' + str(args.top_k):<12} {'MRR@' + str(args.top_k):<12} {tok_label}"
    )
    print("-" * 60)
    for name in result["modes"]:
        m = result[name]
        label = mode_label(name, args.embed_weight if embed_fn else None)
        print(f"{label:<18} {m['recall']:<12} {m['mrr']:<12} {m['avg_tokens']}")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
