"""Tests for nugget_rag.schema.make_chunk (key order is part of the JSON output contract)."""

import json

import pytest

from nugget_rag.schema import make_chunk


def test_small_chunk_key_order_and_arxiv_lookup():
    c = make_chunk(1, "t", 0.5, chunk_index=3)
    assert list(c) == ["paper_id", "chunk_index", "text", "score", "arxiv_id"]
    assert c["arxiv_id"] == "2410.10071"


def test_large_chunk_key_order():
    c = make_chunk(1, "t", 0.5, chunk_indices=[3, 4])
    assert list(c) == ["paper_id", "chunk_indices", "text", "score", "arxiv_id"]


def test_unknown_paper_id_has_no_arxiv_id():
    assert "arxiv_id" not in make_chunk(9999, "t", 0.5, chunk_index=0)


def test_explicit_arxiv_id_first_matches_pdf_layout():
    c = make_chunk(2, "t", 0.0, chunk_indices=[0], arxiv_id="2410.10072", arxiv_id_first=True)
    assert list(c) == ["paper_id", "arxiv_id", "chunk_indices", "text", "score"]
    assert c["arxiv_id"] == "2410.10072"
    assert json.dumps(c).startswith('{"paper_id": 2, "arxiv_id"')


def test_requires_exactly_one_index_key():
    with pytest.raises(ValueError):
        make_chunk(1, "t", 0.0)
    with pytest.raises(ValueError):
        make_chunk(1, "t", 0.0, chunk_index=0, chunk_indices=[0])
