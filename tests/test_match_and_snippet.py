"""Tests for --match normalized (evaluate) and --snippet-length (fetch_papers), #200."""

from __future__ import annotations

import sys

import pytest

from eval.evaluate import (
    _normalize_for_match,
    evaluate,
    main,
    mrr_at_k,
    recall_at_k,
)
from nugget_rag.paper_registry import ARXIV_MAP
from scripts import fetch_papers as fp

PID = ARXIV_MAP["2410.10071"]
# What one extractor yields vs another for the same sentence.
API_TEXT = "Gyuwan Kim at UniversityofCalifornia,SantaBarbara wrote it."
SPAN = "University of California, Santa Barbara"


def test_normalize_drops_space_and_punctuation_keeps_letters_digits():
    assert _normalize_for_match("University of California, Santa-Barbara 0.5") == (
        "universityofcaliforniasantabarbara05"
    )
    assert _normalize_for_match("精度は 0.95 です。") == "精度は095です"


def test_exact_match_misses_but_normalized_hits_across_extractors():
    results = [{"text": API_TEXT}]
    assert recall_at_k(results, [SPAN], "text") is False
    assert recall_at_k(results, [SPAN], "text", normalize=True) is True
    assert mrr_at_k(results, [SPAN], "text") == 0.0
    assert mrr_at_k(results, [SPAN], "text", normalize=True) == 1.0


def test_normalized_match_ignores_empty_spans():
    """A span made only of punctuation normalizes to '' and must not match everything."""
    assert recall_at_k([{"text": "anything"}], ["...", " "], "text", normalize=True) is False


def test_exact_default_is_unchanged():
    assert recall_at_k([{"text": "Alpha Beta"}], ["alpha beta"], "text") is True
    assert recall_at_k([{"text": "AlphaBeta"}], ["alpha beta"], "text") is False


def _chunks():
    return {PID: [{"paper_id": PID, "text": API_TEXT, "nugget": API_TEXT}]}


def _gold():
    return [{"arxiv_id": "2410.10071", "query": "california", "answer_spans": [SPAN]}]


def test_evaluate_match_modes_and_reports_choice():
    exact = evaluate(_chunks(), _gold())
    norm = evaluate(_chunks(), _gold(), match="normalized")
    assert exact["match"] == "exact" and exact["full_chunk"]["recall"] == 0.0
    assert norm["match"] == "normalized" and norm["full_chunk"]["recall"] == 1.0


def test_evaluate_rejects_unknown_match():
    with pytest.raises(ValueError, match="match must be"):
        evaluate(_chunks(), _gold(), match="fuzzy")


def test_main_match_flag(monkeypatch, tmp_path, capsys):
    import json

    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(json.dumps(_chunks()[PID]))
    g.write_text(json.dumps(_gold()))
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c), "--gold", str(g), "--match", "normalized"]
    )
    main()
    assert '"match": "normalized"' in capsys.readouterr().out


# --- fetch_papers: snippet_length reaches the API request -----------------------------------


class _FakeBody:
    def __init__(self):
        self.url = None

    def __call__(self, url, timeout=0, **kw):
        self.url = url
        return '{"results": []}'


def test_fetch_chunks_sends_snippet_length(monkeypatch):
    body = _FakeBody()
    monkeypatch.setattr(fp, "_urlopen_with_retry", body)
    fp.fetch_chunks_for_paper("http://localhost:8020", 1, "q", snippet_length=0)
    assert "snippet_length=0" in body.url


def test_fetch_chunks_default_snippet_length_is_the_api_default(monkeypatch):
    body = _FakeBody()
    monkeypatch.setattr(fp, "_urlopen_with_retry", body)
    fp.fetch_chunks_for_paper("http://localhost:8020", 1, "q")
    assert f"snippet_length={fp.DEFAULT_SNIPPET_LENGTH}" in body.url
    assert fp.DEFAULT_SNIPPET_LENGTH == 200  # unchanged so earlier results stay comparable


def test_fetch_per_query_passes_snippet_length_through(monkeypatch):
    seen = []

    def fake(api, pid, query, limit=20, snippet_length=-1):
        seen.append(snippet_length)
        return []

    monkeypatch.setattr(fp, "fetch_chunks_for_paper", fake)
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["s"]}]
    fp.fetch_per_query("http://x", gold, "small", 512, snippet_length=0)
    assert seen == [0]


def test_cli_rejects_negative_snippet_length(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch_papers.py", "--snippet-length", "-1"])
    with pytest.raises(SystemExit) as exc:
        fp.main()
    assert exc.value.code == 2


def test_main_pool_label_is_echoed(monkeypatch, tmp_path, capsys):
    import json

    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(json.dumps(_chunks()[PID]))
    g.write_text(json.dumps(_gold()))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c), "--gold", str(g), "--pool-label", "unit-test pool"],
    )
    main()
    out = capsys.readouterr().out
    assert '"pool": "unit-test pool"' in out


def test_main_without_pool_label_has_no_pool_key(monkeypatch, tmp_path, capsys):
    import json

    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(json.dumps(_chunks()[PID]))
    g.write_text(json.dumps(_gold()))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--chunks", str(c), "--gold", str(g)])
    main()
    assert '"pool"' not in capsys.readouterr().out
