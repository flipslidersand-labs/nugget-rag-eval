"""#268: the gate compares unrounded recall (hits / n), not the 3-decimal display value."""

from __future__ import annotations

import json
import sys

import pytest

from eval import check_regression
from eval.check_regression import check_baseline
from eval.evaluate import evaluate
from nugget_rag.paper_registry import ARXIV_MAP


def _result(hits: int, n: int) -> dict:
    # 189/199 = 0.94975 -> displayed as 0.95 after round(..., 3)
    shown = round(hits / n, 3)
    return {
        "modes": ["full_chunk", "nugget"],
        "n_queries": n,
        "hits": {"full_chunk": hits, "nugget": hits},
        "full_chunk": {"recall": shown},
        "nugget": {"recall": shown},
    }


def test_evaluate_reports_raw_hit_counts_and_keeps_rounded_recall():
    pid = ARXIV_MAP["2410.10071"]
    chunks = {pid: [{"paper_id": pid, "text": "alpha beta"}]}
    gold = [{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["alpha"]}]
    res = evaluate(chunks, gold)
    assert res["hits"] == {"full_chunk": 1, "nugget": 1}
    assert set(res["full_chunk"]) == {"recall", "mrr", "avg_tokens"}


def _run_main(monkeypatch, capsys, result, extra=()):
    monkeypatch.setattr(check_regression, "evaluate", lambda *a, **k: result)
    monkeypatch.setattr(check_regression, "validate_chunks", lambda c: None)
    monkeypatch.setattr(check_regression, "load_json", lambda p, n: [])
    monkeypatch.setattr(check_regression, "existing_file", lambda p: p)
    monkeypatch.setattr(
        sys, "argv", ["x", "--chunks", "c", "--gold", "g", "--threshold", "0.95", *extra]
    )
    code = 0
    try:
        check_regression.main()
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().out


def test_189_of_199_fails_threshold_0_95(monkeypatch, capsys):
    res = _result(189, 199)
    assert res["full_chunk"]["recall"] == 0.95  # the rounded display value that used to pass
    code, out = _run_main(monkeypatch, capsys, res)
    assert code == 1 and "[FAIL]" in out


def test_exactly_at_threshold_passes(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, _result(190, 200))
    assert code in (0, None) and "[PASS]" in out


def test_result_without_hits_falls_back_to_displayed_recall(monkeypatch, capsys):
    res = {"modes": ["full_chunk"], "full_chunk": {"recall": 0.96}}
    code, _ = _run_main(monkeypatch, capsys, res)
    assert code in (0, None)


def _baseline(tmp_path, recall):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"full_chunk": {"recall": recall}, "nugget": {"recall": recall}}))
    return str(p)


def test_baseline_drop_uses_unrounded_recall(tmp_path):
    # raw 189/199 = 0.94975; baseline 1.0 -> drop 0.05025 > 0.05 (rounded 0.95 gave 0.05)
    failures, _ = check_baseline(_result(189, 199), _baseline(tmp_path, 1.0), max_drop=0.05)
    assert len(failures) == 2


def test_baseline_drop_exactly_max_drop_passes(tmp_path):
    failures, _ = check_baseline(_result(95, 100), _baseline(tmp_path, 1.0), max_drop=0.05)
    assert failures == []


@pytest.mark.parametrize("hits", [0, 50])
def test_baseline_improvement_not_failure(tmp_path, hits):
    failures, _ = check_baseline(_result(hits, 100), _baseline(tmp_path, 0.0), max_drop=0.0)
    assert failures == []
