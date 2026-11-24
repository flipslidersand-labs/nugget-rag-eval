"""Recall@5 regression check for CI.

Runs BM25-only evaluation and fails if Recall@5 drops below the threshold.

Usage:
    python eval/check_regression.py \
        --chunks data/chunks_large.json \
        --gold eval/gold_set.json \
        --threshold 0.95

    # Show which queries caused regression:
    python eval/check_regression.py ... --verbose

    # Baseline mode: fail if recall drops more than --max-drop below the recorded
    # values; --advisory reports the drop as [WARN] and exits 0:
    python eval/check_regression.py \
        --chunks data/chunks_pdf.json \
        --gold eval/gold_set_hard.json \
        --threshold 0 \
        --baseline eval/baseline_hard.json --max-drop 0.05 --advisory

    # Record a baseline from this run (same --top-k/--match/--nugget-window as the later check):
    python eval/check_regression.py --chunks ... --gold ... --write-baseline eval/baseline_x.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from eval.cli_helpers import (
    existing_file,
    float_between_0_1,
    non_negative_int,
    positive_int,
    write_json_atomic,
)
from eval.evaluate import (
    EvalConfig,
    evaluate,
    format_diagnostics,
    load_inputs,
    mode_label,
    pop_diagnostics,
)


def raw_recall(result: dict, mode: str) -> float:
    """Unrounded recall (hits / n_queries); falls back to the displayed value without valid hit counts."""
    all_hits = result.get("hits")
    hits = all_hits.get(mode) if isinstance(all_hits, dict) else None
    n = result.get("n_queries")
    valid = (
        isinstance(hits, int)
        and isinstance(n, int)
        and not isinstance(hits, bool)
        and not isinstance(n, bool)
        and 0 <= hits <= n
        and n > 0
    )
    if not valid:
        return result[mode]["recall"]
    return hits / n


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _same_int(a: object, b: int) -> bool:
    return isinstance(a, int) and not isinstance(a, bool) and a == b


def _same_path(a: object, b: str) -> bool:
    # baseline paths are repo-root relative; --gold is relative to the cwd. Independent of the cwd for the baseline.
    return isinstance(a, str) and (_REPO_ROOT / a).resolve() == Path(b).resolve()


# Marks recall-drop messages; any other baseline failure means "not comparable".
_DROP_MARK = " dropped "
_INCOMPARABLE_NOTE = (
    "[WARN] 比較不能 (not comparable): baseline と今回で top_k/match/nugget_window/gold/n/モード が違うため、"
    "数値の差は意味を持たない"
)


def check_baseline(
    result: dict,
    baseline_path: str,
    max_drop: float,
    top_k: int | None = None,
    gold_path: str | None = None,
) -> tuple[list[str], str]:
    """Compare recall against a recorded baseline.

    Returns (failures, note). A missing baseline file is not a failure: the
    note says the drop check was skipped. Failures list modes whose recall
    fell by more than *max_drop* below the baseline, a baseline sharing no
    mode with this run, and a baseline whose recorded top_k / match /
    nugget_window / gold / n differ from this run. Metadata keys absent from
    the baseline (old format) are not checked (not assumed to be the defaults);
    match / nugget_window are read from ``result["config"]`` and skipped if absent.
    """
    path = Path(baseline_path)
    if not path.is_file():
        return [], f"baseline not found ({baseline_path}); drop check skipped"
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict | list):
            raise TypeError(f"baseline must be a JSON object, got {type(baseline).__name__}")
        base = {
            mode: float(baseline[mode]["recall"]) for mode in result["modes"] if mode in baseline
        }
        for mode, value in base.items():
            # NaN makes `drop > max_drop` always False, so the gate would pass silently (#389).
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{mode} recall must be a finite number in [0, 1], got {value}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"[ERROR] invalid baseline file {baseline_path}: {exc!r}")

    failures = []
    if not base:
        recorded = sorted(baseline) if isinstance(baseline, dict) else []
        failures.append(
            f"baseline shares no mode with this run (baseline modes {recorded}, "
            f"current modes {list(result['modes'])}); recall was not compared"
        )
    if isinstance(baseline, dict):
        # A recorded value that differs (or cannot be compared) means the numbers are not comparable.
        if "top_k" in baseline and top_k is not None and not _same_int(baseline["top_k"], top_k):
            failures.append(f"baseline top_k {baseline['top_k']!r} != current top_k {top_k}")
        config = result.get("config") or {}
        if "match" in baseline and "match" in config and baseline["match"] != config["match"]:
            failures.append(
                f"baseline match {baseline['match']!r} != current match {config['match']}"
            )
        if (
            "nugget_window" in baseline
            and "nugget_window" in config
            and not _same_int(baseline["nugget_window"], config["nugget_window"])
        ):
            failures.append(
                f"baseline nugget_window {baseline['nugget_window']!r} "
                f"!= current nugget_window {config['nugget_window']}"
            )
        if "n" in baseline and not _same_int(baseline["n"], result.get("n_queries")):
            failures.append(f"baseline n {baseline['n']!r} != current n {result.get('n_queries')}")
        if (
            "gold" in baseline
            and gold_path is not None
            and not _same_path(baseline["gold"], gold_path)
        ):
            failures.append(f"baseline gold {baseline['gold']!r} != current gold {gold_path}")

    for mode in base:  # modes without a recorded baseline are not checked yet
        now = result[mode]["recall"]
        # baseline is stored rounded to 3 places, so compare displayed values (#268 covers the threshold only).
        drop = round(base[mode] - now, 3)
        if drop > max_drop:
            failures.append(
                f"{mode} recall {now}{_DROP_MARK}{round(drop, 3)} from baseline {base[mode]} (max-drop {max_drop})"
            )
    return failures, ""


def _baseline_gold_path(gold_path: str) -> str:
    """Repo-root relative POSIX path when *gold_path* is inside the repo, else the absolute path."""
    resolved = Path(gold_path).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_baseline(result: dict, gold_path: str) -> dict:
    """The checkable part of a baseline (config, n, gold, per-mode recall) of one evaluate() run.

    ``_comment`` and ``pool`` are free text and stay hand-written.
    """
    config = result["config"]
    baseline: dict = {
        "gold": _baseline_gold_path(gold_path),
        "top_k": config["top_k"],
        "match": config["match"],
        "nugget_window": config["nugget_window"],
    }
    for mode in result["modes"]:
        baseline[mode] = {"recall": result[mode]["recall"]}
    baseline["n"] = result["n_queries"]
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@5 regression check")
    parser.add_argument(
        "--chunks", required=True, type=existing_file, help="Path to chunks JSON file"
    )
    parser.add_argument(
        "--gold", required=True, type=existing_file, help="Path to gold set JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float_between_0_1,
        default=0.95,
        help="Minimum acceptable Recall@5, in [0, 1] (default: 0.95)",
    )
    parser.add_argument(
        "--top-k", type=positive_int, default=5, help="Top-k for retrieval (default: 5)"
    )
    parser.add_argument(
        "--match",
        choices=["exact", "normalized"],
        default="exact",
        help="Span matching, as in evaluate.py (default: exact). Checked against the baseline.",
    )
    parser.add_argument(
        "--nugget-window",
        type=non_negative_int,
        default=0,
        help="Context sentences on each side of a nugget (default: 0). "
        "Checked against the baseline.",
    )
    parser.add_argument(
        "--write-baseline",
        default=None,
        metavar="PATH",
        help="Write this run's config, n, gold and per-mode recall to PATH and exit 0 "
        "(no gate). Not combinable with --baseline.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="JSON with recorded recall per mode ({'full_chunk': {'recall': x}, 'nugget': {...}}). "
        "Fails when recall drops more than --max-drop below it. Missing file = skipped.",
    )
    parser.add_argument(
        "--max-drop",
        type=float_between_0_1,
        default=0.05,
        help="Allowed recall drop below --baseline (default: 0.05)",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report baseline drops but exit 0 (start as advice, tighten later)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print failing query details to stderr when recall drops below threshold",
    )
    args = parser.parse_args()
    if args.write_baseline and args.baseline:
        parser.error("--write-baseline and --baseline cannot be used together")

    chunks_by_paper, gold = load_inputs(args.chunks, args.gold)

    try:
        result = evaluate(
            chunks_by_paper,
            gold,
            EvalConfig(top_k=args.top_k, match=args.match, nugget_window=args.nugget_window),
        )
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    for line in format_diagnostics(result, verbose=args.verbose):
        print(line, file=sys.stderr)
    pop_diagnostics(result)

    if args.write_baseline:
        write_json_atomic(
            Path(args.write_baseline), build_baseline(result, args.gold), trailing_newline=True
        )
        print(f"[WRITTEN] baseline: {args.write_baseline}")
        return

    threshold = args.threshold

    print(f"Recall@{args.top_k} regression check")
    print(f"  threshold  : {threshold}")
    failed = []
    for name in result["modes"]:
        recall = result[name]["recall"]
        exact = raw_recall(result, name)
        label = mode_label(name)
        print(f"  {label:<10} : {recall}")
        if exact < threshold:
            failed.append(f"{label:<10} Recall@{args.top_k} {exact:.5f} < threshold {threshold}")

    drops: list[str] = []
    if args.baseline:
        drops, note = check_baseline(
            result, args.baseline, args.max_drop, top_k=args.top_k, gold_path=args.gold
        )
        if note:
            print(f"  [NOTE] {note}")

    if failed:
        print("\n[FAIL] Recall regression detected:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)

    if drops:
        label = "[WARN]" if args.advisory else "[FAIL]"
        print(f"\n{label} Baseline check failed:")
        for msg in drops:
            print(f"  - {msg}")
        if not args.advisory:
            sys.exit(1)
        if any(_DROP_MARK not in msg for msg in drops):
            print(f"  {_INCOMPARABLE_NOTE}")
        print("  (advisory: exit 0)")
        return

    print(f"\n[PASS] Recall@{args.top_k} >= {threshold} for both modes.")


if __name__ == "__main__":
    main()
