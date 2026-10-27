"""Tests for scripts/make_eval_report.py."""

from __future__ import annotations

import json
import sys

import pytest

from eval.evaluate import build_chunks_index
from nugget_rag.paper_registry import ARXIV_MAP
from scripts import make_eval_report as mr

PID1, PID2 = ARXIV_MAP["2410.10071"], ARXIV_MAP["2508.11836"]
CHUNKS = [
    {
        "paper_id": PID1,
        "arxiv_id": "2410.10071",
        "chunk_indices": [0],
        "text": "Alpha one is here. Beta two is there. Gamma three follows.",
        "score": 1.0,
    },
    {
        "paper_id": PID2,
        "arxiv_id": "2508.11836",
        "chunk_indices": [0],
        "text": "Alpha one appears here too. Delta four ends it.",
        "score": 1.0,
    },
]
GOLD = [{"arxiv_id": "2410.10071", "query": "alpha one", "answer_spans": ["Gamma three follows"]}]


def test_parse_named_variants():
    assert mr.parse_named("B=data/x.json", "p") == ("B", "data/x.json", None)
    assert mr.parse_named("C=data/x.json:normalized", "p") == ("C", "data/x.json", "normalized")
    # a colon that is not a match suffix stays part of the path
    assert mr.parse_named("D=C:/tmp/x.json", "p") == ("D", "C:/tmp/x.json", None)
    for bad in ("nopath", "=x.json", "N="):
        with pytest.raises(ValueError):
            mr.parse_named(bad, "p")


def test_run_matrix_has_one_row_per_combination():
    pools = {"P": (build_chunks_index(CHUNKS), "exact")}
    rows = mr.run_matrix(pools, {"G": GOLD}, windows=[0, 1])
    assert len(rows) == 4  # 1 pool x 1 gold x 2 scopes x 2 windows
    assert {(r["scope"], r["window"]) for r in rows} == {
        ("paper", 0),
        ("paper", 1),
        ("all", 0),
        ("all", 1),
    }
    assert all(r["pool"] == "P" and r["gold"] == "G" for r in rows)


def test_all_papers_scope_really_widens_the_search_range():
    """The answer lives only in another paper's chunk: invisible in 'paper' scope, found in 'all'."""
    gold = [
        {"arxiv_id": "2410.10071", "query": "delta four", "answer_spans": ["Delta four ends it"]}
    ]
    pools = {"P": (build_chunks_index(CHUNKS), "exact")}
    paper = mr.run_matrix(pools, {"G": gold}, [0], scopes=("paper",))[0]
    allp = mr.run_matrix(pools, {"G": gold}, [0], scopes=("all",))[0]
    assert paper["full_chunk"]["recall"] == 0.0
    assert allp["full_chunk"]["recall"] == 1.0
    assert allp["n_queries"] == paper["n_queries"] == 1  # same gold, only the range differs


def test_markdown_has_both_tables_and_every_row():
    pools = {"P": (build_chunks_index(CHUNKS), "exact")}
    rows = mr.run_matrix(pools, {"G": GOLD}, [0, 1])
    md = mr.format_markdown(rows)
    assert md.count("| P | G |") == 2 * len(rows)  # once per row in each table
    assert "正解論文のみ" in md and "全論文" in md
    assert "McNemar p" in md
    assert ("できる" in md) or ("できない" in md)  # every comparison row states a verdict


def test_main_writes_markdown_and_json(monkeypatch, tmp_path, capsys):
    c, g, j = tmp_path / "c.json", tmp_path / "g.json", tmp_path / "r.json"
    c.write_text(json.dumps(CHUNKS))
    g.write_text(json.dumps(GOLD))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rep",
            "--pool",
            f"P={c}:normalized",
            "--gold",
            f"G={g}",
            "--windows",
            "0",
            "--scopes",
            "paper",
            "--json",
            str(j),
        ],
    )
    mr.main()
    assert "| P | G | 正解論文のみ | 0 |" in capsys.readouterr().out
    rows = json.loads(j.read_text(encoding="utf-8"))
    assert len(rows) == 1 and rows[0]["match"] == "normalized"


def test_main_bad_spec_exits(monkeypatch, tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps(GOLD))
    monkeypatch.setattr(sys, "argv", ["rep", "--pool", "nopath", "--gold", f"G={g}"])
    with pytest.raises(SystemExit) as exc:
        mr.main()
    assert "NAME=PATH" in str(exc.value.code)
