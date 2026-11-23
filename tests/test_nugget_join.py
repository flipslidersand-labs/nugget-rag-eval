"""Nugget text must not create strings that are absent from the source chunk (#301)."""

import pytest

from eval.evaluate import _normalize_for_match, recall_at_k
from nugget_rag.retriever import _SEAM, retrieve_full_chunk, retrieve_nuggets

# Sentences 0 and 2 are the only ones sharing query terms; sentence 1 is filler between them.
GAP_TEXT = "We use KV. Filler sentence about nothing here at all. Cache is reduced by quantization."
GAP_QUERY = "KV cache reduced quantization"
GAP_SPAN = ["KV cache"]

CJK_TEXT = "手法を提案する。精度が向上した。"
CJK_QUERY = "手法 精度"
CJK_SPAN = ["提案する。精度"]


def _fake_embed(texts):
    return [[1.0, 0.0] for _ in texts]


def _pick(text, query, **kw):
    chunks = [{"paper_id": 1, "text": text}]
    return chunks, retrieve_nuggets(chunks, query, top_k=1, **kw)[0]["nugget"]


@pytest.mark.parametrize("embed", [False, True], ids=["bm25", "hybrid"])
def test_non_adjacent_sentences_do_not_form_a_normalized_hit(embed):
    kw = {"embed_fn": _fake_embed, "embed_weight": 0.0001} if embed else {}
    chunks, nugget = _pick(GAP_TEXT, GAP_QUERY, nuggets_per_chunk=2, **kw)
    # Guard: the picks really are sentences 0 and 2, so this is not vacuous.
    assert "We use KV." in nugget and "Cache is reduced" in nugget
    assert "Filler" not in nugget
    assert not recall_at_k(retrieve_full_chunk(chunks, GAP_QUERY), GAP_SPAN, "text", True)
    assert not recall_at_k([{"nugget": nugget}], GAP_SPAN, "nugget", True)


def test_non_adjacent_sentences_do_not_form_an_exact_hit():
    _, nugget = _pick(GAP_TEXT, GAP_QUERY, nuggets_per_chunk=2)
    assert not recall_at_k([{"nugget": nugget}], ["KV. Cache"], "nugget", False)


@pytest.mark.parametrize("embed", [False, True], ids=["bm25", "hybrid"])
def test_adjacent_cjk_sentences_keep_the_original_text(embed):
    kw = {"embed_fn": _fake_embed, "embed_weight": 0.0001} if embed else {}
    chunks, nugget = _pick(CJK_TEXT, CJK_QUERY, nuggets_per_chunk=2, **kw)
    assert nugget == CJK_TEXT
    assert recall_at_k(retrieve_full_chunk(chunks, CJK_QUERY), CJK_SPAN, "text", False)
    assert recall_at_k([{"nugget": nugget}], CJK_SPAN, "nugget", False)


def test_adjacent_latin_sentences_still_joined_with_a_space():
    text = "Alpha one. Beta two. Gamma three. Delta four."
    _, nugget = _pick(text, "beta gamma", nuggets_per_chunk=2)
    assert nugget == "Beta two. Gamma three."


def test_window_neighbours_are_adjacent_not_seams():
    text = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five."
    _, nugget = _pick(text, "gamma", nuggets_per_chunk=1, window=1)
    assert nugget == "Beta two. Gamma three. Delta four."
    assert _SEAM not in nugget


def test_duplicate_sentences_do_not_fabricate_adjacency():
    text = "Same line. Other filler. Same line. KV cache quantization."
    _, nugget = _pick(text, "same line kv cache quantization", nuggets_per_chunk=2)
    assert nugget == f"Same line. {_SEAM}KV cache quantization."


def test_seam_survives_match_normalisation_and_keeps_word_count():
    assert _normalize_for_match(_SEAM) != ""
    _, nugget = _pick(GAP_TEXT, GAP_QUERY, nuggets_per_chunk=2)
    # Old output was "We use KV. Cache is reduced by quantization." (8 words); seam adds none.
    assert len(nugget.split()) == 8
