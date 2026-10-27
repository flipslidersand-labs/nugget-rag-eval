"""Apply a human review of the hard gold set (#194) to eval/gold_set_hard.json.

The review is the text copied from the review sheet, one verdict per gold item:

    #194 目視確認(40件版) ○30 △2 ×8 未判定0
    1. 2410.10071 ○
    2. 2508.11836 △ 直す点のメモ
    3. 2508.11845 ×  理由

Numbers are 1-based positions in the gold file the sheet was built from. Items
marked × are dropped; ○ and △ are kept (△ items are listed so they can be fixed
by hand). By default nothing is written: the script prints what it would do.

Usage:
    python scripts/apply_gold_review.py review.txt              # dry run
    python scripts/apply_gold_review.py review.txt --write      # rewrite the gold file
    python scripts/apply_gold_review.py review.txt --out new.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from eval.cli_helpers import existing_file, load_json

VERDICTS = {"○": "ok", "△": "fix", "×": "ng", "未判定": "none"}
LINE = re.compile(r"^\s*(\d+)\.\s+(\S+)\s+(○|△|×|未判定)\s*(.*?)\s*$")


class ReviewError(ValueError):
    """The review text does not match the gold file; nothing must be applied."""


def parse_review(text: str) -> dict[int, tuple[str, str, str]]:
    """Return {n: (arxiv_id, verdict, note)} from the sheet's copied text."""
    out: dict[int, tuple[str, str, str]] = {}
    for raw in text.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        n = int(m[1])
        if n in out:
            raise ReviewError(f"item {n} appears more than once")
        out[n] = (m[2], VERDICTS[m[3]], m[4])
    if not out:
        raise ReviewError("no verdict lines found (expected lines like '3. 2508.11845 ○')")
    return out


def apply_review(
    gold: list[dict], review: dict[int, tuple[str, str, str]], allow_unjudged: bool = False
) -> tuple[list[dict], dict[str, list[tuple[int, dict, str]]]]:
    """Return (new_gold, report). Raises ReviewError on any mismatch."""
    if set(review) != set(range(1, len(gold) + 1)):
        missing = sorted(set(range(1, len(gold) + 1)) - set(review))
        extra = sorted(set(review) - set(range(1, len(gold) + 1)))
        raise ReviewError(
            f"review covers {len(review)} items but gold has {len(gold)} "
            f"(missing {missing[:8]}, unexpected {extra[:8]}); was the sheet built from this gold file?"
        )
    report: dict[str, list[tuple[int, dict, str]]] = {"ok": [], "fix": [], "ng": [], "none": []}
    for n, item in enumerate(gold, 1):
        arxiv, verdict, note = review[n]
        if arxiv != item["arxiv_id"]:
            raise ReviewError(
                f"item {n}: review says {arxiv} but gold[{n - 1}] is {item['arxiv_id']}; "
                "the sheet and the gold file are out of sync"
            )
        report[verdict].append((n, item, note))
    if report["none"] and not allow_unjudged:
        nums = [n for n, _, _ in report["none"]]
        raise ReviewError(
            f"{len(nums)} item(s) still unjudged: {nums}; judge them or pass --allow-unjudged"
        )
    dropped = {n for n, _, _ in report["ng"]}
    kept = [item for n, item in enumerate(gold, 1) if n not in dropped]
    if not kept:
        raise ReviewError("every item would be dropped; refusing to write an empty gold set")
    return kept, report


def summarize(report: dict[str, list[tuple[int, dict, str]]], total: int, kept: int) -> str:
    lines = [f"gold {total} items -> {kept} kept ({len(report['ng'])} dropped)"]
    for key, title in (
        ("ng", "dropped (×)"),
        ("fix", "kept but needs a fix (△)"),
        ("none", "unjudged (kept)"),
    ):
        if report[key]:
            lines.append(f"\n{title}:")
            for n, item, note in report[key]:
                lines.append(
                    f"  {n}. {item['arxiv_id']}  {item['query'][:70]}"
                    + (f"  — {note}" if note else "")
                )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("review", type=existing_file, help="text copied from the review sheet")
    ap.add_argument("--gold", default="eval/gold_set_hard.json", type=existing_file)
    ap.add_argument("--out", type=Path, help="write the reviewed gold here instead of in place")
    ap.add_argument("--write", action="store_true", help="rewrite --gold in place")
    ap.add_argument("--allow-unjudged", action="store_true")
    args = ap.parse_args()

    gold = load_json(args.gold, "--gold")
    try:
        review = parse_review(Path(args.review).read_text(encoding="utf-8"))
        kept, report = apply_review(gold, review, args.allow_unjudged)
    except ReviewError as exc:
        sys.exit(f"[ERROR] {exc}")

    print(summarize(report, len(gold), len(kept)))
    target = args.out or (Path(args.gold) if args.write else None)
    if target is None:
        print("\n(dry run: pass --write or --out to save)")
        return
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    print(f"\nwrote {len(kept)} items to {target}")


if __name__ == "__main__":
    main()
