"""Tests for scripts/analyze_nugget_failures.py (#201)."""

from __future__ import annotations

import json
import sys

from eval.evaluate import build_chunks_index
from nugget_rag.paper_registry import ARXIV_MAP
from scripts import analyze_nugget_failures as af

PID = ARXIV_MAP["2410.10071"]
FILLER = " ".join(f"Filler sentence number {i} about nothing." for i in range(6))


def _by(*texts):
    return build_chunks_index(
        [
            {
                "paper_id": PID,
                "arxiv_id": "2410.10071",
                "chunk_indices": [i],
                "text": t,
                "score": 1.0,
            }
            for i, t in enumerate(texts)
        ]
    )


def _gold(query, span):
    return [{"arxiv_id": "2410.10071", "query": query, "answer_spans": [span]}]


def test_D_hit_when_the_answer_sentence_matches_the_query():
    by = _by("The zebra crossing rule is key. " + FILLER)
    classes, _ = af.classify_all(by, _gold("zebra crossing rule", "zebra crossing rule is key"))
    assert classes == {"D": 1}


def test_A_when_no_top_chunk_has_the_span():
    by = _by("Completely unrelated text here. " + FILLER)
    classes, _ = af.classify_all(by, _gold("zebra crossing", "zebra crossing rule is key"))
    assert classes == {"A": 1}


def test_B_when_the_span_straddles_two_sentences():
    by = _by("The zebra crossing. Rule is key. " + FILLER)
    classes, _ = af.classify_all(by, _gold("zebra crossing rule", "crossing. Rule is"))
    assert classes == {"B": 1}


def test_C_when_the_answer_sentence_is_ranked_below_the_cut_and_reports_its_rank():
    # query words only appear in the first sentences; the answer sentence shares none
    text = "Alpha beta gamma one. Alpha beta two. Alpha beta three. Alpha beta four. Zeta omega closes."
    by = _by(text)
    classes, c_rank = af.classify_all(by, _gold("alpha beta", "Zeta omega closes"), per_chunk=3)
    assert classes == {"C": 1} and c_rank == {"<=5": 1}


def test_only_the_gold_paper_is_searched():
    other = ARXIV_MAP["2508.11836"]
    chunks = [
        {
            "paper_id": other,
            "arxiv_id": "2508.11836",
            "chunk_indices": [0],
            "text": "The zebra crossing rule is key.",
            "score": 1.0,
        },
        {
            "paper_id": PID,
            "arxiv_id": "2410.10071",
            "chunk_indices": [0],
            "text": FILLER,
            "score": 1.0,
        },
    ]
    classes, _ = af.classify_all(
        build_chunks_index(chunks), _gold("zebra crossing", "zebra crossing rule")
    )
    assert classes == {"A": 1}  # the answer exists, but in another paper


def test_normalized_match_changes_the_class_for_glued_text():
    by = _by("TheZebraCrossingRule is key. " + FILLER)
    gold = _gold("zebra crossing rule", "The Zebra Crossing Rule is key")
    assert af.classify_all(by, gold, normalize=False)[0] == {"A": 1}
    assert af.classify_all(by, gold, normalize=True)[0] == {"D": 1}


def test_sweep_recall_never_drops_when_k_grows_and_window_adds_context():
    text = "Alpha beta gamma one. Alpha beta two. Alpha beta three. Alpha beta four. Zeta omega closes."
    by = _by(text)
    rows = af.sweep(by, _gold("alpha beta", "Zeta omega closes"), ks=(1, 3, 5), windows=(0, 1))
    r = {(x["k"], x["window"]): x["recall"] for x in rows}
    assert r[(1, 0)] <= r[(3, 0)] <= r[(5, 0)] == 1.0
    assert r[(1, 0)] == 0.0 and r[(5, 0)] == 1.0  # the sweep actually changes the outcome
    assert all(r[(k, 1)] >= r[(k, 0)] for k in (1, 3, 5))  # a window only adds sentences


def test_format_report_lists_all_classes():
    out = af.format_report({"A": 2, "C": 1, "D": 3}, {"<=5": 1}, 6)
    assert "n=6" in out and "A 検索の失敗" in out and "D 成功" in out and "<=5 1" in out


def test_main_prints_report_and_sweep(monkeypatch, tmp_path, capsys):
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(
        json.dumps(
            [
                {
                    "paper_id": PID,
                    "arxiv_id": "2410.10071",
                    "chunk_indices": [0],
                    "text": "The zebra crossing rule is key. " + FILLER,
                    "score": 1.0,
                }
            ]
        )
    )
    g.write_text(json.dumps(_gold("zebra crossing rule", "zebra crossing rule is key")))
    monkeypatch.setattr(sys, "argv", ["af", "--chunks", str(c), "--gold", str(g), "--sweep"])
    af.main()
    out = capsys.readouterr().out
    assert "D 成功: 1" in out and "k  window  Recall" in out
