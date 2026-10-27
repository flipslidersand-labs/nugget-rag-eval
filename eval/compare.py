"""Paired comparison of full-chunk vs nugget retrieval (#195).

Pure functions over per-query hit flags; stdlib only. The decision rule is
fixed in the issue before measuring: the two modes are *distinguishable* only
if the exact McNemar p-value is below 0.05 AND the 95% bootstrap interval of
the recall difference excludes 0.
"""

from __future__ import annotations

import random
from math import comb

ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
CRITERION = "distinguishable iff mcnemar_p < 0.05 and recall_diff_ci95 excludes 0"


def contingency(full_hits: list[bool], nugget_hits: list[bool]) -> dict[str, int]:
    """Count per-query outcomes: both / full_only / nugget_only / neither."""
    if len(full_hits) != len(nugget_hits):
        raise ValueError("full_hits and nugget_hits must have the same length")
    both = sum(f and g for f, g in zip(full_hits, nugget_hits))
    full_only = sum(f and not g for f, g in zip(full_hits, nugget_hits))
    nugget_only = sum(g and not f for f, g in zip(full_hits, nugget_hits))
    return {
        "both": both,
        "full_only": full_only,
        "nugget_only": nugget_only,
        "neither": len(full_hits) - both - full_only - nugget_only,
    }


def mcnemar_exact_p(full_only: int, nugget_only: int) -> float:
    """Two-sided exact binomial test on the discordant pairs (p0 = 0.5)."""
    k = full_only + nugget_only
    if k == 0:
        return 1.0
    tail = sum(comb(k, i) for i in range(min(full_only, nugget_only) + 1))
    return min(1.0, 2 * tail / 2**k)


def bootstrap_diff_ci(
    full_hits: list[bool],
    nugget_hits: list[bool],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """95% percentile interval of Recall(full) - Recall(nugget), resampling queries."""
    n = len(full_hits)
    if n == 0:
        return None
    diffs = [int(f) - int(g) for f, g in zip(full_hits, nugget_hits)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return lo, hi


def compare(
    full_hits: list[bool],
    nugget_hits: list[bool],
    full_tokens: float,
    nugget_tokens: float,
) -> dict:
    """Build the comparison block reported by evaluate()."""
    n = len(full_hits)
    counts = contingency(full_hits, nugget_hits)
    full_recall = sum(full_hits) / n if n else 0.0
    nugget_recall = sum(nugget_hits) / n if n else 0.0
    p = mcnemar_exact_p(counts["full_only"], counts["nugget_only"])
    ci = bootstrap_diff_ci(full_hits, nugget_hits)
    distinguishable = bool(ci is not None and p < ALPHA and (ci[0] > 0 or ci[1] < 0))
    return {
        "counts": counts,
        "recall_diff": round(full_recall - nugget_recall, 3),
        "recall_diff_ci95": [round(ci[0], 3), round(ci[1], 3)] if ci else None,
        "mcnemar_p": round(p, 4),
        "token_ratio": round(nugget_tokens / full_tokens, 3) if full_tokens else None,
        "recall_retained": round(nugget_recall / full_recall, 3) if full_recall else None,
        "distinguishable": distinguishable,
        "criterion": CRITERION,
    }
