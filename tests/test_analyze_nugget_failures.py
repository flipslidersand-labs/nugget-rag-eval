"""Tests for scripts/analyze_nugget_failures.py (#201)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from eval.evaluate import build_chunks_index, evaluate
from nugget_rag.paper_registry import ARXIV_MAP
from nugget_rag.retriever import NuggetSelection, retrieve_full_chunk
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


@pytest.mark.parametrize(
    "rank, bucket",
    [(5, "<=5"), (6, "<=8"), (8, "<=8"), (9, ">8")],
)
def test_C_rank_bucket_boundaries(rank, bucket):
    # rank-1 decoy sentences share the query words (score > 0); the answer sentence shares none
    # (score 0) and comes last, so its BM25 rank among its chunk's sentences is exactly `rank`.
    decoys = [f"Alpha beta number{i}." for i in range(rank - 1)]
    text = " ".join([*decoys, "Zeta omega closes."])
    by = _by(text)
    gold = _gold("alpha beta", "Zeta omega closes")

    # guard: the synthetic chunk really puts the answer sentence at the intended rank
    cls, got_rank = af.classify(by[PID], "alpha beta", gold[0]["answer_spans"], 5, 3, False)
    assert (cls, got_rank) == ("C", rank)

    classes, c_rank = af.classify_all(by, gold, per_chunk=3)
    assert classes == {"C": 1} and c_rank == {bucket: 1}


def _gold_spans(query, *spans):
    return [{"arxiv_id": "2410.10071", "query": query, "answer_spans": list(spans)}]


# Span 1 straddles a sentence boundary in B_CHUNK; span 2 sits inside one sentence in D_CHUNK.
# (The straddling sentences share no query word and three decoys fill the nugget slots, so
# evaluate's nugget text does not join them either.)
B_CHUNK = (
    "Zebra crossing rule one. Zebra crossing rule two. Zebra crossing rule three. "
    "Alpha beta gamma. Delta epsilon zeta. " + FILLER
)
D_CHUNK = "The zebra crossing rule is key. " + FILLER
B_D_GOLD = _gold_spans("zebra crossing rule", "gamma. Delta", "zebra crossing rule is key")


def test_D_when_a_later_holder_chunk_hits_even_if_the_first_is_B():
    by = _by(B_CHUNK, D_CHUNK)
    top = retrieve_full_chunk(by[PID], "zebra crossing rule", 5)
    # guard: both chunks hold a span and the B chunk ranks first (the old holders[0])
    assert [c["text"] for c in top] == [B_CHUNK, D_CHUNK]
    classes, _ = af.classify_all(by, B_D_GOLD)
    assert classes == {"D": 1}


def test_C_beats_B_and_reports_the_best_rank_over_all_holders():
    # chunk 0: answer sentence at rank 6 (C). chunk 1: answer sentence at rank 4 (C).
    # chunk 2: span straddles sentences (B). Result: C with the smallest rank, 4.
    def chunk(answer, n_decoys):
        return " ".join([*(f"Alpha beta number{i}." for i in range(n_decoys)), answer])

    by = _by(
        chunk("Zeta omega closes.", 5),
        chunk("Theta sigma closes.", 3),
        "Alpha beta. Straddles here. " + FILLER,
    )
    gold = _gold_spans("alpha beta", "Zeta omega closes", "Theta sigma closes", "beta. Straddles")
    cls, rank = af.classify(by[PID], "alpha beta", gold[0]["answer_spans"], 5, 3, False)
    assert (cls, rank) == ("C", 4)
    classes, c_rank = af.classify_all(by, gold, per_chunk=3)
    assert classes == {"C": 1} and c_rank == {"<=5": 1}


def test_B_when_every_holder_straddles_and_none_has_a_sentence_holding_the_span():
    by = _by(B_CHUNK, B_CHUNK.replace("Alpha", "Omega"))
    classes, _ = af.classify_all(by, _gold_spans("zebra crossing rule", "gamma. Delta"))
    assert classes == {"B": 1}


def test_D_count_matches_evaluate_nugget_hits():
    # Mixed outcomes, including a multi-holder item whose first holder is B (#327).
    gold = [
        *B_D_GOLD,
        *_gold("zebra crossing rule", "zebra crossing rule is key"),  # D
        *_gold("zebra crossing", "no such answer anywhere"),  # A
        *_gold_spans("zebra crossing rule", "gamma. Delta"),  # B
        *_gold("alpha beta", "Zeta omega closes"),  # C
    ]
    by = _by(
        B_CHUNK,
        D_CHUNK,
        "Alpha beta gamma one. Alpha beta two. Alpha beta three. Alpha beta four. "
        "Zeta omega closes.",
    )
    classes, _ = af.classify_all(by, gold, top_k=5, per_chunk=3)
    result = evaluate(by, gold, top_k=5)  # BM25-only: no embed_fn, nugget_window=0
    assert classes["D"] == result["hits"]["nugget"]
    assert classes == {"D": 2, "A": 1, "B": 1, "C": 1}


def test_D_count_matches_evaluate_on_the_committed_fixtures():
    root = Path(__file__).parent / "fixtures"
    by = build_chunks_index(json.loads((root / "chunks_test.json").read_text()))
    gold = json.loads((root / "gold_test.json").read_text())
    for normalize in (False, True):
        classes, _ = af.classify_all(by, gold, normalize=normalize)
        result = evaluate(by, gold, match="normalized" if normalize else "exact")
        assert classes["D"] == result["hits"]["nugget"]


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


def test_classify_uses_the_retriever_selection(monkeypatch):
    # #248: the verdict follows retriever.select_nuggets, not a private copy of its logic.
    by = _by(D_CHUNK)
    spans = ["zebra crossing rule is key"]
    assert af.classify(by[PID], "zebra crossing rule", spans, 5, 3, False) == ("D", None)
    real = af.select_nuggets

    def pick_nothing(*a, **kw):
        return [NuggetSelection(s.chunk, s.sentences, []) for s in real(*a, **kw)]

    monkeypatch.setattr(af, "select_nuggets", pick_nothing)
    cls, rank = af.classify(by[PID], "zebra crossing rule", spans, 5, 3, False)
    assert cls == "C" and rank == 1
