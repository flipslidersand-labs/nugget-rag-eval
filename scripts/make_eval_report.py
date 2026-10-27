"""Print the README evaluation tables from measurements, so numbers are never typed by hand.

Runs evaluate() for every (pool x gold x nugget window x search scope) and prints
markdown tables: Recall/MRR/tokens per mode, and the paired full-vs-nugget comparison.

Usage:
    python scripts/make_eval_report.py \\
        --pool "B=data/chunks_pdf.json:exact" \\
        --pool "C=data/chunks_api_full.json:normalized" \\
        --gold "既存19=eval/gold_set.json" --gold "難問=eval/gold_set_hard.json" \\
        --windows 0 1 --json report.json

A pool is NAME=PATH[:MATCH] with MATCH exact (default) or normalized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.cli_helpers import load_json, non_negative_int
from eval.evaluate import build_chunks_index, evaluate, validate_chunks, validate_gold


def parse_named(spec: str, what: str) -> tuple[str, str, str | None]:
    """'NAME=PATH' or 'NAME=PATH:MATCH' -> (name, path, match|None)."""
    if "=" not in spec:
        raise ValueError(f"{what} must look like NAME=PATH, got {spec!r}")
    name, rest = spec.split("=", 1)
    match = None
    if ":" in rest and rest.rsplit(":", 1)[1] in ("exact", "normalized"):
        rest, match = rest.rsplit(":", 1)
    if not name or not rest:
        raise ValueError(f"{what} must look like NAME=PATH, got {spec!r}")
    return name, rest, match


def run_matrix(
    pools: dict[str, tuple[dict, str]],
    golds: dict[str, list[dict]],
    windows: list[int],
    scopes: tuple[str, ...] = ("paper", "all"),
) -> list[dict]:
    """One row per (pool, gold, window, scope); each row is the evaluate() result plus its keys."""
    rows = []
    for pname, (by_paper, match) in pools.items():
        for gname, gold in golds.items():
            for scope in scopes:
                chunks = by_paper
                if scope == "all":
                    pooled = [c for cs in by_paper.values() for c in cs]
                    chunks = {k: pooled for k in by_paper}
                for w in windows:
                    res = evaluate(chunks, gold, match=match, nugget_window=w)
                    rows.append(
                        {
                            "pool": pname,
                            "gold": gname,
                            "scope": scope,
                            "window": w,
                            "match": match,
                            **res,
                        }
                    )
    return rows


def _cell(r: dict, mode: str) -> str:
    m = r[mode]
    return f"{m['recall']:.3f} / {m['mrr']:.3f} / {m['avg_tokens']:.1f}"


def format_markdown(rows: list[dict]) -> str:
    out = [
        "### Recall@5 / MRR@5 / 平均語数",
        "",
        "| プール | ゴールド | 範囲 | 窓 | 件数 | full-chunk | nugget |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        scope = "正解論文のみ" if r["scope"] == "paper" else "全論文"
        out.append(
            f"| {r['pool']} | {r['gold']} | {scope} | {r['window']} | {r['n_queries']} | "
            f"{_cell(r, 'full_chunk')} | {_cell(r, 'nugget')} |"
        )
    out += [
        "",
        "### full-chunk と nugget の差（paired）",
        "",
        "| プール | ゴールド | 範囲 | 窓 | Recall 差と 95% 区間 | McNemar p | 語数/full | 保持率 | 判別 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["comparison"]
        ci = c["recall_diff_ci95"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        scope = "正解論文のみ" if r["scope"] == "paper" else "全論文"
        out.append(
            f"| {r['pool']} | {r['gold']} | {scope} | {r['window']} | {c['recall_diff']:.3f} {ci_s} | "
            f"{c['mcnemar_p']} | {c['token_ratio']} | {c['recall_retained']} | "
            f"{'できる' if c['distinguishable'] else 'できない'} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pool", action="append", required=True, help="NAME=PATH[:exact|normalized]")
    ap.add_argument("--gold", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--windows", nargs="+", type=non_negative_int, default=[0, 1])
    ap.add_argument("--scopes", nargs="+", choices=["paper", "all"], default=["paper", "all"])
    ap.add_argument("--json", type=Path, help="also write the raw rows here")
    args = ap.parse_args()

    try:
        pools = {}
        for spec in args.pool:
            name, path, match = parse_named(spec, "--pool")
            chunks = load_json(path, f"--pool {name}")
            validate_chunks(chunks)
            pools[name] = (build_chunks_index(chunks), match or "exact")
        golds = {}
        for spec in args.gold:
            name, path, _ = parse_named(spec, "--gold")
            gold = load_json(path, f"--gold {name}")
            validate_gold(gold)
            golds[name] = gold
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    rows = run_matrix(pools, golds, args.windows, tuple(args.scopes))
    print(format_markdown(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
