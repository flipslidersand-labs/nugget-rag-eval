"""main() input-validation and --match paths of scripts/analyze_nugget_failures.py (#246)."""

from __future__ import annotations

import json
import sys

import pytest

from nugget_rag.paper_registry import ARXIV_MAP
from scripts import analyze_nugget_failures as af

PID = ARXIV_MAP["2410.10071"]
FILLER = " ".join(f"Filler sentence number {i} about nothing." for i in range(6))
GLUED = "TheZebraCrossingRule is key. " + FILLER


def _files(tmp_path, text=GLUED, span="The Zebra Crossing Rule is key"):
    chunks = [
        {
            "paper_id": PID,
            "arxiv_id": "2410.10071",
            "chunk_indices": [0],
            "text": text,
            "score": 1.0,
        }
    ]
    gold = [{"arxiv_id": "2410.10071", "query": "zebra crossing rule", "answer_spans": [span]}]
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(json.dumps(chunks), encoding="utf-8")
    g.write_text(json.dumps(gold), encoding="utf-8")
    return c, g


def _run(monkeypatch, c, g, *extra):
    monkeypatch.setattr(sys, "argv", ["af", "--chunks", str(c), "--gold", str(g), *extra])
    af.main()


def test_main_match_normalized_hits_glued_text(monkeypatch, tmp_path, capsys):
    c, g = _files(tmp_path)
    _run(monkeypatch, c, g, "--match", "normalized")
    assert "D 成功: 1" in capsys.readouterr().out


def test_main_match_exact_is_the_default_and_misses_glued_text(monkeypatch, tmp_path, capsys):
    c, g = _files(tmp_path)
    _run(monkeypatch, c, g)
    assert "A 検索の失敗（根拠チャンクが上位に入らない）: 1" in capsys.readouterr().out


def test_main_normalized_sweep_uses_normalize(monkeypatch, tmp_path, capsys):
    c, g = _files(tmp_path)
    _run(monkeypatch, c, g, "--match", "normalized", "--sweep")
    rows = capsys.readouterr().out.split("k  window")[1].splitlines()[1:]
    assert rows and all(r.split()[2] == "1.0" for r in rows if r.strip())


@pytest.mark.parametrize("which", ["chunks", "gold"])
def test_main_exits_on_corrupt_json(monkeypatch, tmp_path, which):
    c, g = _files(tmp_path)
    {"chunks": c, "gold": g}[which].write_text("[oops", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, c, g)
    assert f"--{which} file is not valid JSON" in str(exc.value)


def test_main_exits_on_chunks_schema_error(monkeypatch, tmp_path):
    c, g = _files(tmp_path)
    c.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, c, g)
    assert str(exc.value).startswith("[ERROR] chunks must be a JSON array")


def test_main_exits_on_gold_schema_error(monkeypatch, tmp_path):
    c, g = _files(tmp_path)
    g.write_text(json.dumps([{"arxiv_id": "2410.10071", "query": "q", "answer_spans": []}]))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, c, g)
    assert str(exc.value).startswith("[ERROR] gold[0]")


@pytest.mark.parametrize(
    "extra", [["--top-k", "0"], ["--nuggets-per-chunk", "0"], ["--match", "fuzzy"]]
)
def test_main_rejects_bad_arguments(monkeypatch, tmp_path, extra, capsys):
    c, g = _files(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, c, g, *extra)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err and extra[0] in err


def test_main_rejects_missing_file(monkeypatch, tmp_path, capsys):
    _, g = _files(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path / "nope.json", g)
    assert exc.value.code == 2
    assert "file not found" in capsys.readouterr().err


def test_main_nuggets_per_chunk_is_passed_through(monkeypatch, tmp_path, capsys):
    text = "Alpha beta gamma one. Alpha beta two. Alpha beta three. Alpha beta four. Zeta omega closes."
    c, g = _files(tmp_path, text=text, span="Zeta omega closes")
    g.write_text(
        json.dumps(
            [
                {
                    "arxiv_id": "2410.10071",
                    "query": "alpha beta",
                    "answer_spans": ["Zeta omega closes"],
                }
            ]
        )
    )
    _run(monkeypatch, c, g, "--nuggets-per-chunk", "1")
    one = capsys.readouterr().out
    _run(monkeypatch, c, g, "--nuggets-per-chunk", "5")
    five = capsys.readouterr().out
    assert "C 抽出の失敗（チャンクは入ったが根拠文が選ばれない）: 1" in one
    assert "D 成功: 1" in five


def test_main_exits_when_gold_and_chunks_do_not_match(monkeypatch, tmp_path):
    c, g = _files(tmp_path)
    other = [
        {
            **json.loads(c.read_text())[0],
            "paper_id": ARXIV_MAP["2508.11836"],
            "arxiv_id": "2508.11836",
        }
    ]
    c.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, c, g)
    assert str(exc.value.code).startswith("[ERROR] no gold paper key found in chunks")
