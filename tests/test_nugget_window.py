"""Tests for the nugget neighbour-sentence window (#201)."""

from __future__ import annotations

import sys

import pytest

from eval.evaluate import evaluate, main
from nugget_rag.retriever import retrieve_nuggets
from nugget_rag.scorer import top_nuggets

SENTS = [
    "Alpha opens the paper.",
    "Background about nothing.",
    "The zebra crossing rule is the key finding.",
    "Follow up detail after it.",
    "Unrelated closing remark.",
]
QUERY = "zebra crossing rule"


def test_window_zero_is_unchanged():
    assert top_nuggets(QUERY, SENTS, top_k=1) == top_nuggets(QUERY, SENTS, top_k=1, window=0)
    assert top_nuggets(QUERY, SENTS, top_k=1) == [SENTS[2]]


def test_window_one_adds_neighbours_in_document_order():
    assert top_nuggets(QUERY, SENTS, top_k=1, window=1) == SENTS[1:4]


def test_window_clamps_at_edges():
    sents = ["zebra crossing rule here.", "second.", "third."]
    assert top_nuggets(QUERY, sents, top_k=1, window=2) == sents


def test_window_has_no_duplicates_when_picks_overlap():
    out = top_nuggets("zebra crossing rule follow", SENTS, top_k=2, window=1)
    assert len(out) == len(set(out))
    assert out == sorted(out, key=SENTS.index)


def test_window_on_empty_sentences():
    assert top_nuggets(QUERY, [], top_k=3, window=1) == []


def test_retrieve_nuggets_passes_window_through():
    chunks = [{"paper_id": 1, "text": " ".join(SENTS), "score": 1.0}]
    plain = retrieve_nuggets(chunks, QUERY, top_k=1, nuggets_per_chunk=1)[0]["nugget"]
    wide = retrieve_nuggets(chunks, QUERY, top_k=1, nuggets_per_chunk=1, window=1)[0]["nugget"]
    assert plain == SENTS[2]
    assert wide == " ".join(SENTS[1:4])


def test_evaluate_window_raises_nugget_recall_without_touching_full_chunk():
    spans = ["Follow up detail"]
    # build_chunks_index keys by the registry's internal paper_id (1 for 2410.10071)
    chunks = {1: [{"paper_id": 1, "arxiv_id": "2410.10071", "text": " ".join(SENTS), "score": 1.0}]}
    gold = [{"query": QUERY, "arxiv_id": "2410.10071", "answer_spans": spans}]
    base = evaluate(chunks, gold, top_k=1)
    wide = evaluate(chunks, gold, top_k=1, nugget_window=1)
    assert base["nugget"]["recall"] == 0.0
    assert wide["nugget"]["recall"] == 1.0
    assert base["full_chunk"] == wide["full_chunk"]


def test_main_rejects_negative_window(monkeypatch, tmp_path):
    c = tmp_path / "c.json"
    g = tmp_path / "g.json"
    c.write_text("[]")
    g.write_text("[]")
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c), "--gold", str(g), "--nugget-window", "-1"]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
