"""Tests for eval/compare.py (paired full-chunk vs nugget statistics, #195)."""

from __future__ import annotations

import pytest

from eval.compare import (
    bootstrap_diff_ci,
    compare,
    contingency,
    mcnemar_exact_p,
)
from eval.evaluate import evaluate
from nugget_rag.paper_registry import ARXIV_MAP


def test_contingency_counts_and_sum():
    full = [True, True, False, False, True]
    nug = [True, False, True, False, False]
    c = contingency(full, nug)
    assert c == {"both": 1, "full_only": 2, "nugget_only": 1, "neither": 1}
    assert sum(c.values()) == len(full)


def test_contingency_length_mismatch_raises():
    with pytest.raises(ValueError):
        contingency([True], [True, False])


def test_mcnemar_exact_known_values():
    # no discordant pairs -> p = 1
    assert mcnemar_exact_p(0, 0) == 1.0
    # 5 vs 0: 2 * C(5,0) / 2^5 = 2/32
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    # 10 vs 0: 2/1024
    assert mcnemar_exact_p(10, 0) == pytest.approx(2 / 1024)
    # symmetric
    assert mcnemar_exact_p(3, 7) == mcnemar_exact_p(7, 3)
    # perfectly balanced -> capped at 1
    assert mcnemar_exact_p(4, 4) == 1.0


def test_bootstrap_is_deterministic_and_brackets_point_estimate():
    full = [True] * 30 + [False] * 10
    nug = [True] * 10 + [False] * 30
    a = bootstrap_diff_ci(full, nug)
    b = bootstrap_diff_ci(full, nug)
    assert a == b  # fixed seed
    point = (30 - 10) / 40
    assert a[0] <= point <= a[1]


def test_bootstrap_empty_returns_none():
    assert bootstrap_diff_ci([], []) is None


def test_bootstrap_identical_modes_has_zero_width_at_zero():
    hits = [True, False, True, False]
    assert bootstrap_diff_ci(hits, hits) == (0.0, 0.0)


def test_compare_distinguishable_when_full_clearly_better():
    full = [True] * 21 + [False] * 19
    nug = [True] * 8 + [False] * 32  # nested: every nugget hit is also a full hit
    out = compare(full, nug, full_tokens=280.0, nugget_tokens=70.0)
    assert out["counts"]["nugget_only"] == 0
    assert out["recall_diff"] == pytest.approx(0.325)
    assert out["distinguishable"] is True
    assert out["mcnemar_p"] < 0.05
    assert out["recall_diff_ci95"][0] > 0
    assert out["token_ratio"] == 0.25
    assert out["recall_retained"] == pytest.approx(0.381, abs=1e-3)


def test_compare_not_distinguishable_when_identical():
    hits = [True, False, True, True]
    out = compare(hits, hits, 100.0, 100.0)
    assert out["distinguishable"] is False
    assert out["mcnemar_p"] == 1.0
    assert out["recall_diff"] == 0.0


def test_compare_not_distinguishable_with_tiny_sample():
    # 3 vs 0 discordant: p = 0.25, so the rule must refuse to call it
    out = compare([True, True, True], [False, False, False], 10.0, 10.0)
    assert out["mcnemar_p"] == pytest.approx(0.25)
    assert out["distinguishable"] is False


def test_compare_handles_empty_and_zero_recall():
    out = compare([], [], 0.0, 0.0)
    assert out["recall_diff_ci95"] is None
    assert out["distinguishable"] is False
    assert out["token_ratio"] is None
    zero = compare([False, False], [False, False], 10.0, 5.0)
    assert zero["recall_retained"] is None


def test_evaluate_includes_comparison_block_and_keeps_existing_keys():
    pid = ARXIV_MAP["2410.10071"]
    chunks = {pid: [{"paper_id": pid, "text": "alpha beta gamma. delta epsilon."}]}
    gold = [{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["alpha"]}]
    res = evaluate(chunks, gold)
    assert {"full_chunk", "nugget", "latency_ms", "comparison"} <= set(res)
    cmp_ = res["comparison"]
    assert sum(cmp_["counts"].values()) == res["n_queries"]
    assert "criterion" in cmp_
