"""Tests for the MODES table in eval/evaluate.py (#181)."""

from __future__ import annotations

import json
import sys

import pytest

from eval import check_regression
from eval import evaluate as ev
from nugget_rag.paper_registry import ARXIV_MAP

PID = ARXIV_MAP["2410.10071"]
CHUNKS = {PID: [{"paper_id": PID, "text": "alpha beta gamma. delta epsilon zeta."}]}
GOLD = [{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["alpha"]}]


def test_result_lists_modes_in_table_order_and_keeps_legacy_keys():
    res = ev.evaluate(CHUNKS, GOLD)
    assert res["modes"] == [m.name for m in ev.MODES] == ["full_chunk", "nugget"]
    for name in res["modes"]:
        assert set(res[name]) == {"recall", "mrr", "avg_tokens"}
        assert name in res["latency_ms"]


def test_mode_label():
    assert ev.mode_label("full_chunk") == "full-chunk"
    assert ev.mode_label("nugget") == "nugget"
    assert ev.mode_label("nugget", 0.3) == "nugget(e0.3)"
    assert ev.mode_label("full_chunk", 0.3) == "full-chunk"  # weight only labels nugget


def test_mode_without_registered_retriever_is_rejected(monkeypatch):
    """Adding a MODES entry without wiring its retriever must fail loudly, not silently skip."""
    extra = ev.RetrievalMode("rerank", "text", "rerank")
    monkeypatch.setattr(ev, "MODES", (*ev.MODES, extra))
    with pytest.raises(AssertionError, match="out of sync"):
        ev.evaluate(CHUNKS, GOLD)


def test_check_baseline_skips_modes_without_a_baseline():
    result = {
        "modes": ["full_chunk", "nugget", "rerank"],
        "full_chunk": {"recall": 0.5},
        "nugget": {"recall": 0.5},
        "rerank": {"recall": 0.0},  # no baseline recorded for it yet -> not checked
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "b.json"
        b.write_text(json.dumps({"full_chunk": {"recall": 0.5}, "nugget": {"recall": 0.5}}))
        failures, note = check_regression.check_baseline(result, str(b), max_drop=0.05)
    assert failures == [] and note == ""


def test_check_regression_prints_every_mode_with_aligned_labels(monkeypatch, tmp_path, capsys):
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(json.dumps([{"paper_id": PID, "text": "alpha beta gamma", "nugget": "alpha"}]))
    g.write_text(json.dumps(GOLD))
    monkeypatch.setattr(sys, "argv", ["check_regression.py", "--chunks", str(c), "--gold", str(g)])
    check_regression.main()
    out = capsys.readouterr().out
    assert "  full-chunk : 1.0" in out
    assert "  nugget     : 1.0" in out
