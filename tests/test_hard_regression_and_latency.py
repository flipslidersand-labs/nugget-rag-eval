"""Tests for the baseline drop check (#202) and latency reporting in evaluate()."""

from __future__ import annotations

import json
import sys

import pytest

from eval import check_regression
from eval.check_regression import check_baseline
from eval.evaluate import _percentile, evaluate
from nugget_rag.paper_registry import ARXIV_MAP

RESULT = {
    "modes": ["full_chunk", "nugget"],  # evaluate() always reports its mode list (#181)
    "full_chunk": {"recall": 0.5},
    "nugget": {"recall": 0.2},
}


def _baseline(tmp_path, full=0.525, nugget=0.2):
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({"full_chunk": {"recall": full}, "nugget": {"recall": nugget}}))
    return str(p)


def test_baseline_within_max_drop_passes(tmp_path):
    failures, note = check_baseline(RESULT, _baseline(tmp_path), max_drop=0.05)
    assert failures == [] and note == ""


def test_baseline_drop_over_max_fails(tmp_path):
    failures, _ = check_baseline(RESULT, _baseline(tmp_path, full=0.7), max_drop=0.05)
    assert len(failures) == 1 and "full_chunk" in failures[0]


def test_baseline_improvement_is_not_a_failure(tmp_path):
    failures, _ = check_baseline(RESULT, _baseline(tmp_path, full=0.1, nugget=0.0), max_drop=0.0)
    assert failures == []


def test_baseline_missing_file_is_skipped(tmp_path):
    failures, note = check_baseline(RESULT, str(tmp_path / "nope.json"), max_drop=0.05)
    assert failures == [] and "skipped" in note


def test_baseline_invalid_file_exits(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"full_chunk": 1}')
    with pytest.raises(SystemExit) as exc:
        check_baseline(RESULT, str(p), max_drop=0.05)
    assert "invalid baseline" in str(exc.value.code)


def _write_main_inputs(tmp_path):
    pid = ARXIV_MAP["2410.10071"]
    chunks = [{"paper_id": pid, "text": "alpha beta gamma", "nugget": "alpha beta gamma"}]
    gold = [{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["alpha beta"]}]
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(json.dumps(chunks))
    g.write_text(json.dumps(gold))
    return c, g


def _run_main(monkeypatch, tmp_path, *extra):
    c, g = _write_main_inputs(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["check_regression.py", "--chunks", str(c), "--gold", str(g), *extra]
    )
    check_regression.main()


def test_main_baseline_drop_fails_unless_advisory(monkeypatch, tmp_path, capsys):
    base = _baseline(tmp_path, full=1.0, nugget=1.0)  # actual recall is 1.0 -> no drop
    _run_main(monkeypatch, tmp_path, "--baseline", base)
    assert "[PASS]" in capsys.readouterr().out


def test_main_baseline_regression_exit_1(monkeypatch, tmp_path):
    # force a recorded baseline above reality by requiring a gold span that is absent
    c, g = _write_main_inputs(tmp_path)
    g.write_text(
        json.dumps([{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["zzz"]}])
    )
    base = _baseline(tmp_path, full=0.9, nugget=0.9)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_regression.py",
            "--chunks",
            str(c),
            "--gold",
            str(g),
            "--threshold",
            "0",
            "--baseline",
            base,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        check_regression.main()
    assert exc.value.code == 1


def test_main_baseline_regression_advisory_exits_0(monkeypatch, tmp_path, capsys):
    c, g = _write_main_inputs(tmp_path)
    g.write_text(
        json.dumps([{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["zzz"]}])
    )
    base = _baseline(tmp_path, full=0.9, nugget=0.9)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_regression.py",
            "--chunks",
            str(c),
            "--gold",
            str(g),
            "--threshold",
            "0",
            "--baseline",
            base,
            "--advisory",
        ],
    )
    check_regression.main()  # must not raise
    out = capsys.readouterr().out
    assert "[WARN]" in out and "advisory" in out


def test_percentile_nearest_rank():
    assert _percentile([], 95) == 0.0
    assert _percentile([5.0], 50) == 5.0
    vals = [float(i) for i in range(1, 101)]
    assert _percentile(vals, 50) == 50.0
    assert _percentile(vals, 95) == 95.0


def test_evaluate_reports_latency_without_embed():
    pid = ARXIV_MAP["2410.10071"]
    chunks = {pid: [{"paper_id": pid, "text": "alpha beta gamma"}]}
    gold = [{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["alpha"]}]
    lat = evaluate(chunks, gold)["latency_ms"]
    assert set(lat) == {"full_chunk", "nugget"}
    assert lat["nugget"]["p95"] >= lat["nugget"]["p50"] >= 0


def test_evaluate_separates_embed_time():
    pid = ARXIV_MAP["2410.10071"]
    chunks = {pid: [{"paper_id": pid, "text": "alpha beta gamma. delta epsilon."}]}
    gold = [{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["alpha"]}]
    calls = []

    def fake_embed(texts):
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    lat = evaluate(chunks, gold, embed_fn=fake_embed)["latency_ms"]
    assert calls, "embed_fn must be called through the timing wrapper"
    assert "nugget_embed_calls" in lat
