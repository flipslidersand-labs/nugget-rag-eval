"""Tests for eval/check_regression.py — main() PASS / FAIL paths."""
from __future__ import annotations

import json
import sys

import pytest

from eval.evaluate import ARXIV_MAP


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _passing_data(tmp_path):
    """Chunks and gold where both full-chunk and nugget recall = 1.0."""
    paper_id = ARXIV_MAP["2410.10071"]  # == 1
    chunks = [{"paper_id": paper_id, "text": "correct answer here", "nugget": "correct answer here"}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["correct answer here"]}]
    c_path, g_path = tmp_path / "chunks.json", tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    return c_path, g_path


def _failing_data(tmp_path):
    """Chunks and gold where recall = 0 (no answer match)."""
    chunks = [{"paper_id": 1, "text": "totally unrelated content", "nugget": "totally unrelated"}]
    gold = [{"paper_id": 99, "query": "q", "answer_spans": ["missing answer"]}]
    c_path, g_path = tmp_path / "chunks.json", tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    return c_path, g_path


# ── import check_regression.main ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _ensure_eval_on_path(monkeypatch):
    """Make sure eval/ dir is importable for check_regression's relative import."""
    import pathlib

    eval_dir = str(pathlib.Path(__file__).parent.parent / "eval")
    if eval_dir not in sys.path:
        monkeypatch.syspath_prepend(eval_dir)
    yield


def _run_main(monkeypatch, args: list[str]):
    """Import and call check_regression.main() with given CLI args."""
    import importlib

    import eval.check_regression as cr_module

    importlib.reload(cr_module)
    monkeypatch.setattr(sys, "argv", ["check_regression.py", *args])
    cr_module.main()


# ── PASS tests ────────────────────────────────────────────────────────────────

def test_main_pass_exits_zero(tmp_path, monkeypatch, capsys):
    c_path, g_path = _passing_data(tmp_path)
    _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.5"])
    out = capsys.readouterr().out
    assert "[PASS]" in out


def test_main_pass_prints_recall_values(tmp_path, monkeypatch, capsys):
    c_path, g_path = _passing_data(tmp_path)
    _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.5"])
    out = capsys.readouterr().out
    assert "full-chunk" in out
    assert "nugget" in out
    assert "threshold" in out


def test_main_pass_custom_threshold(tmp_path, monkeypatch, capsys):
    c_path, g_path = _passing_data(tmp_path)
    _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.1"])
    out = capsys.readouterr().out
    assert "[PASS]" in out
    assert "0.1" in out


def test_main_pass_custom_top_k(tmp_path, monkeypatch, capsys):
    c_path, g_path = _passing_data(tmp_path)
    _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.5", "--top-k", "3"])
    out = capsys.readouterr().out
    assert "[PASS]" in out
    assert "Recall@3" in out


# ── FAIL tests ────────────────────────────────────────────────────────────────

def test_main_fail_exits_one_on_low_recall(tmp_path, monkeypatch):
    c_path, g_path = _failing_data(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.95"])
    assert exc_info.value.code == 1


def test_main_fail_prints_fail_message(tmp_path, monkeypatch, capsys):
    c_path, g_path = _failing_data(tmp_path)
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.95"])
    out = capsys.readouterr().out
    assert "[FAIL]" in out
    assert "Recall regression" in out


def test_main_fail_mentions_both_failing_modes(tmp_path, monkeypatch, capsys):
    c_path, g_path = _failing_data(tmp_path)
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.95"])
    out = capsys.readouterr().out
    assert "full-chunk" in out
    assert "nugget" in out


def test_main_fail_threshold_zero_still_passes(tmp_path, monkeypatch, capsys):
    """threshold=0.0 means any recall (even 0.0) passes."""
    c_path, g_path = _failing_data(tmp_path)
    _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.0"])
    out = capsys.readouterr().out
    assert "[PASS]" in out


def test_main_default_threshold_is_0_95(tmp_path, monkeypatch, capsys):
    """Running without --threshold uses 0.95 default and prints it."""
    c_path, g_path = _passing_data(tmp_path)
    _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path)])
    out = capsys.readouterr().out
    assert "0.95" in out


# ── arxiv_id-keyed chunks (PR #78 追加テスト) ────────────────────────────────

def test_arxiv_id_only_chunks_do_not_raise_key_error(tmp_path, monkeypatch):
    """arxiv_id のみを持つチャンク（paper_id なし）でも KeyError が出ない。"""
    chunks = [{"arxiv_id": "2410.10071", "text": "answer text", "nugget": "answer text"}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["answer"]}]
    c_path, g_path = tmp_path / "c.json", tmp_path / "g.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    try:
        _run_main(monkeypatch, ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.0"])
    except SystemExit as e:
        assert e.code != "KeyError", "KeyError が発生した"
