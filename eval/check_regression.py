"""Recall@5 regression check for CI.

Runs BM25-only evaluation and fails if Recall@5 drops below the threshold.

Usage:
    python eval/check_regression.py \
        --chunks data/chunks_large.json \
        --gold eval/gold_set.json \
        --threshold 0.95

    # Show which queries caused regression:
    python eval/check_regression.py ... --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.evaluate import (
    _existing_file,
    _float_between_0_1,
    _load_json,
    _positive_int,
    build_chunks_index,
    evaluate,
    validate_chunks,
)


def check_baseline(result: dict, baseline_path: str, max_drop: float) -> tuple[list[str], str]:
    """Compare recall against a recorded baseline.

    Returns (failures, note). A missing baseline file is not a failure: the
    note says the drop check was skipped. Failures list modes whose recall
    fell by more than *max_drop* below the baseline.
    """
    path = Path(baseline_path)
    if not path.is_file():
        return [], f"baseline not found ({baseline_path}); drop check skipped"
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        base = {mode: float(baseline[mode]["recall"]) for mode in ("full_chunk", "nugget")}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"[ERROR] invalid baseline file {baseline_path}: {exc!r}")

    failures = []
    for mode in ("full_chunk", "nugget"):
        now = result[mode]["recall"]
        drop = round(base[mode] - now, 3)
        if drop > max_drop:
            failures.append(
                f"{mode} recall {now} dropped {drop} from baseline {base[mode]} (max-drop {max_drop})"
            )
    return failures, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@5 regression check")
    parser.add_argument(
        "--chunks", required=True, type=_existing_file, help="Path to chunks JSON file"
    )
    parser.add_argument(
        "--gold", required=True, type=_existing_file, help="Path to gold set JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=_float_between_0_1,
        default=0.95,
        help="Minimum acceptable Recall@5, in [0, 1] (default: 0.95)",
    )
    parser.add_argument(
        "--top-k", type=_positive_int, default=5, help="Top-k for retrieval (default: 5)"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="JSON with recorded recall per mode ({'full_chunk': {'recall': x}, 'nugget': {...}}). "
        "Fails when recall drops more than --max-drop below it. Missing file = skipped.",
    )
    parser.add_argument(
        "--max-drop",
        type=_float_between_0_1,
        default=0.05,
        help="Allowed recall drop below --baseline (default: 0.05)",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report baseline drops but exit 0 (ADR-006: start as advice, tighten later)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print failing query details to stderr when recall drops below threshold",
    )
    args = parser.parse_args()

    chunks_data: list[dict] = _load_json(args.chunks, "--chunks")
    gold: list[dict] = _load_json(args.gold, "--gold")

    try:
        validate_chunks(chunks_data)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    chunks_by_paper = build_chunks_index(chunks_data)

    try:
        result = evaluate(chunks_by_paper, gold, top_k=args.top_k, verbose=args.verbose)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

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

    drops: list[str] = []
    if args.baseline:
        drops, note = check_baseline(result, args.baseline, args.max_drop)
        if note:
            print(f"  [NOTE] {note}")

    if failed:
        print("\n[FAIL] Recall regression detected:")
        for msg in failed:
            print(f"  - {msg}")
        sys.exit(1)

    if drops:
        label = "[WARN]" if args.advisory else "[FAIL]"
        print(f"\n{label} Recall dropped below baseline:")
        for msg in drops:
            print(f"  - {msg}")
        if not args.advisory:
            sys.exit(1)
        print("  (advisory: exit 0)")
        return

    print(f"\n[PASS] Recall@{args.top_k} >= {threshold} for both modes.")


if __name__ == "__main__":
    main()
