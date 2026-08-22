"""Tests for eval/check_regression.py — paper_id / arxiv_id 両対応."""
from __future__ import annotations
import json, sys
import pytest
from eval.evaluate import ARXIV_MAP

def _write(path, data): path.write_text(json.dumps(data))

def _run(monkeypatch, args):
    import importlib, pathlib
    eval_dir = str(pathlib.Path(__file__).parent.parent / "eval")
    if eval_dir not in sys.path:
        monkeypatch.syspath_prepend(eval_dir)
    import eval.check_regression as cr
    importlib.reload(cr)
    monkeypatch.setattr(sys, "argv", ["check_regression.py", *args])
    cr.main()

# --- paper_id のみのチャンク（従来形式）---

def test_paper_id_only_chunks_pass(tmp_path, monkeypatch, capsys):
    paper_id = ARXIV_MAP["2410.10071"]
    chunks = [{"paper_id": paper_id, "text": "KV cache answer", "nugget": "KV cache answer"}]
    gold = [{"arxiv_id": "2410.10071", "query": "KV", "answer_spans": ["KV cache"]}]
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    _write(c, chunks); _write(g, gold)
    _run(monkeypatch, ["--chunks", str(c), "--gold", str(g), "--threshold", "0.5"])
    assert "[PASS]" in capsys.readouterr().out

# --- arxiv_id を持つチャンク（新形式）---

def test_arxiv_id_chunks_do_not_raise_key_error(tmp_path, monkeypatch):
    """arxiv_id のみのチャンクでも KeyError にならない。"""
    chunks = [{"arxiv_id": "2410.10071", "text": "answer text", "nugget": "answer text"}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["answer"]}]
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    _write(c, chunks); _write(g, gold)
    # KeyError にならなければ OK（recall が 0 でも threshold を 0 にして exit 0 に）
    try:
        _run(monkeypatch, ["--chunks", str(c), "--gold", str(g), "--threshold", "0.0"])
    except SystemExit as e:
        assert e.code != "KeyError", "KeyError が発生した"

# --- 閾値を下回ったら exit 1 ---

def test_fail_exits_one(tmp_path, monkeypatch):
    chunks = [{"paper_id": 99, "text": "unrelated", "nugget": "unrelated"}]
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["missing"]}]
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    _write(c, chunks); _write(g, gold)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["--chunks", str(c), "--gold", str(g), "--threshold", "0.95"])
    assert exc.value.code == 1

# --- 閾値 0.0 なら常に pass ---

def test_threshold_zero_always_pass(tmp_path, monkeypatch, capsys):
    chunks = [{"paper_id": 1, "text": "x", "nugget": "x"}]
    gold = [{"paper_id": 99, "query": "q", "answer_spans": ["missing"]}]
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    _write(c, chunks); _write(g, gold)
    _run(monkeypatch, ["--chunks", str(c), "--gold", str(g), "--threshold", "0.0"])
    assert "[PASS]" in capsys.readouterr().out
