"""Tests for evaluate.py — verbose path and main() CLI."""
from __future__ import annotations

import json
import sys

import pytest

from eval.evaluate import ARXIV_MAP, avg_tokens, evaluate

# ── verbose output path ──────────────────────────────────────────────────────

def _chunks_for(paper_id: int, text: str, nugget: str) -> dict[int, list[dict]]:
    return {paper_id: [{"text": text, "nugget": nugget}]}


def test_evaluate_verbose_prints_miss_to_stderr(capsys):
    """verbose=True with a nugget miss must print to stderr."""
    chunks_by_paper = {1: [{"text": "irrelevant", "nugget": "irrelevant"}]}
    gold = [{"paper_id": 1, "query": "missing span", "answer_spans": ["target answer"]}]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=True)
    captured = capsys.readouterr()
    assert "Nugget Recall Misses" in captured.err


def test_evaluate_verbose_no_miss_is_silent(capsys):
    """verbose=True with all nuggets hitting must not print to stderr."""
    chunks_by_paper = {1: [{"text": "target answer text", "nugget": "target answer"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["target answer"]}]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=True)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_evaluate_verbose_false_is_silent_on_miss(capsys):
    """verbose=False (default) must never print to stderr."""
    chunks_by_paper = {1: [{"text": "irrelevant", "nugget": "irrelevant"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["missing"]}]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=False)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_evaluate_verbose_caps_at_5_misses(capsys):
    """verbose output shows at most 5 misses even if more exist."""
    chunks_by_paper = {i: [{"text": "x", "nugget": "x"}] for i in range(1, 9)}
    gold = [
        {"paper_id": i, "query": f"q{i}", "answer_spans": ["miss"]}
        for i in range(1, 9)
    ]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=True)
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("  [")]
    assert len(lines) <= 5


# ── avg_tokens estimator ─────────────────────────────────────────────────────

def test_avg_tokens_chars_mode():
    results = [{"text": "abcd"}]  # 4 chars → 4/4 = 1.0
    assert avg_tokens(results, "text", estimator="chars") == 1.0


def test_avg_tokens_words_mode():
    results = [{"text": "one two three"}]
    assert avg_tokens(results, "text", estimator="words") == 3.0


def test_avg_tokens_empty_results():
    assert avg_tokens([], "text", estimator="words") == 0.0
    assert avg_tokens([], "text", estimator="chars") == 0.0


# ── evaluate() chars estimator ───────────────────────────────────────────────

def test_evaluate_chars_estimator_produces_result():
    chunks_by_paper = {1: [{"text": "eight chr", "nugget": "eight chr"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["eight chr"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5, estimator="chars")
    assert result["full_chunk"]["avg_tokens"] > 0


def test_evaluate_empty_gold_returns_zeros():
    result = evaluate({}, [], top_k=5)
    assert result["n_queries"] == 0
    assert result["full_chunk"]["recall"] == 0
    assert result["nugget"]["recall"] == 0
    assert result["full_chunk"]["mrr"] == 0


# ── main() CLI via subprocess ─────────────────────────────────────────────────

def _write_json(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture()
def minimal_data(tmp_path):
    """Return (chunks_path, gold_path) with one passing entry."""
    paper_id = ARXIV_MAP["2410.10071"]  # == 1
    chunks = [{"paper_id": paper_id, "text": "unique answer phrase", "nugget": "unique answer phrase"}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["unique answer phrase"]}]
    c_path = tmp_path / "chunks.json"
    g_path = tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    return c_path, g_path


def test_main_bm25_only_exits_zero(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)])
    main()  # must not raise
    out = capsys.readouterr().out
    assert "full-chunk" in out
    assert "nugget" in out


def test_main_chars_estimator(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--token-estimator", "chars"],
    )
    main()
    out = capsys.readouterr().out
    assert "chars" in out


def test_main_verbose_flag(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--verbose"],
    )
    main()
    out = capsys.readouterr().out
    assert "full-chunk" in out


def test_main_top_k_flag(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--top-k", "3"],
    )
    main()
    out = capsys.readouterr().out
    assert "Recall@3" in out


def test_main_outputs_json(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)])
    main()
    out = capsys.readouterr().out
    # stdout ends with a JSON block starting at the first "{"
    json_part = out[out.find("{"):]
    parsed = json.loads(json_part)
    assert "full_chunk" in parsed
    assert "nugget" in parsed
    assert "n_queries" in parsed


def test_main_paper_id_chunk_key(tmp_path, monkeypatch, capsys):
    """Chunks with paper_id (int) are found by gold with arxiv_id resolved via ARXIV_MAP."""
    from eval.evaluate import main

    paper_id = ARXIV_MAP["2410.10071"]  # == 1
    chunks = [{"paper_id": paper_id, "text": "answer text", "nugget": "answer text"}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["answer text"]}]
    c_path, g_path = tmp_path / "chunks.json", tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)])
    main()
    out = capsys.readouterr().out
    json_part = out[out.find("{"):]
    parsed = json.loads(json_part)
    assert parsed["full_chunk"]["recall"] == 1.0
