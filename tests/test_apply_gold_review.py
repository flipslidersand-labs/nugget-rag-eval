"""Tests for scripts/apply_gold_review.py (#194)."""

from __future__ import annotations

import json
import sys

import pytest

from scripts import apply_gold_review as ar

GOLD = [
    {"arxiv_id": "2410.10071", "query": "q one", "answer_spans": ["a"]},
    {"arxiv_id": "2508.11836", "query": "q two", "answer_spans": ["b"]},
    {"arxiv_id": "2508.11845", "query": "q three", "answer_spans": ["c"]},
    {"arxiv_id": "2509.25673", "query": "q four", "answer_spans": ["d"]},
]
SHEET = """#194 目視確認(40件版) ○2 △1 ×1 未判定0
1. 2410.10071 ○
2. 2508.11836 △ 言い換えが弱い
3. 2508.11845 ×  同内容が付録にもある
4. 2509.25673 ○
"""


def test_parse_review_reads_verdicts_and_notes():
    r = ar.parse_review(SHEET)
    assert r[1] == ("2410.10071", "ok", "")
    assert r[2] == ("2508.11836", "fix", "言い換えが弱い")
    assert r[3] == ("2508.11845", "ng", "同内容が付録にもある")
    assert len(r) == 4  # the header line is ignored


def test_parse_review_rejects_duplicates_and_empty():
    with pytest.raises(ar.ReviewError, match="more than once"):
        ar.parse_review("1. 2410.10071 ○\n1. 2410.10071 ×\n")
    with pytest.raises(ar.ReviewError, match="no verdict lines"):
        ar.parse_review("just some text\n")


def test_apply_drops_only_x_and_keeps_fix_items():
    kept, report = ar.apply_review(GOLD, ar.parse_review(SHEET))
    assert [g["query"] for g in kept] == ["q one", "q two", "q four"]
    assert [n for n, _, _ in report["ng"]] == [3]
    assert [n for n, _, _ in report["fix"]] == [2]


def test_apply_refuses_when_gold_and_review_are_out_of_sync():
    wrong = SHEET.replace("3. 2508.11845", "3. 2601.10849")
    with pytest.raises(ar.ReviewError, match="out of sync"):
        ar.apply_review(GOLD, ar.parse_review(wrong))


def test_apply_refuses_a_partial_review():
    partial = "\n".join(SHEET.splitlines()[:3])
    with pytest.raises(ar.ReviewError, match="covers 2 items but gold has 4"):
        ar.apply_review(GOLD, ar.parse_review(partial))


def test_unjudged_blocks_unless_allowed():
    sheet = SHEET.replace("4. 2509.25673 ○", "4. 2509.25673 未判定")
    with pytest.raises(ar.ReviewError, match="unjudged"):
        ar.apply_review(GOLD, ar.parse_review(sheet))
    kept, report = ar.apply_review(GOLD, ar.parse_review(sheet), allow_unjudged=True)
    assert len(kept) == 3 and [n for n, _, _ in report["none"]] == [4]  # unjudged items stay


def test_refuses_to_empty_the_gold_set():
    sheet = "\n".join(f"{i}. {g['arxiv_id']} ×" for i, g in enumerate(GOLD, 1))
    with pytest.raises(ar.ReviewError, match="empty gold"):
        ar.apply_review(GOLD, ar.parse_review(sheet))


def _files(tmp_path):
    g, r = tmp_path / "gold.json", tmp_path / "review.txt"
    g.write_text(json.dumps(GOLD, ensure_ascii=False))
    r.write_text(SHEET, encoding="utf-8")
    return g, r


def test_main_dry_run_does_not_write(monkeypatch, tmp_path, capsys):
    g, r = _files(tmp_path)
    before = g.read_text()
    monkeypatch.setattr(sys, "argv", ["apply", str(r), "--gold", str(g)])
    ar.main()
    assert g.read_text() == before
    out = capsys.readouterr().out
    assert "4 items -> 3 kept" in out and "dry run" in out and "言い換えが弱い" in out


def test_main_write_rewrites_in_place_atomically(monkeypatch, tmp_path):
    g, r = _files(tmp_path)
    monkeypatch.setattr(sys, "argv", ["apply", str(r), "--gold", str(g), "--write"])
    ar.main()
    assert [x["query"] for x in json.loads(g.read_text(encoding="utf-8"))] == [
        "q one",
        "q two",
        "q four",
    ]
    assert not g.with_suffix(".tmp").exists()


def test_main_out_leaves_the_original(monkeypatch, tmp_path):
    g, r = _files(tmp_path)
    out = tmp_path / "new.json"
    before = g.read_text()
    monkeypatch.setattr(sys, "argv", ["apply", str(r), "--gold", str(g), "--out", str(out)])
    ar.main()
    assert g.read_text() == before and len(json.loads(out.read_text(encoding="utf-8"))) == 3


def test_main_error_exits_without_writing(monkeypatch, tmp_path):
    g, r = _files(tmp_path)
    r.write_text(SHEET.replace("3. 2508.11845", "3. 2601.10849"), encoding="utf-8")
    before = g.read_text()
    monkeypatch.setattr(sys, "argv", ["apply", str(r), "--gold", str(g), "--write"])
    with pytest.raises(SystemExit) as exc:
        ar.main()
    assert "out of sync" in str(exc.value.code) and g.read_text() == before
