"""Tests for scripts/fetch_papers.py.

Covers: _sanitize_query, combine_large_chunks,
        fetch_chunks_for_paper (mocked), fetch_per_query (mocked).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from scripts.fetch_papers import (
    ARXIV_TO_PAPER_ID,
    PAPER_ID_TO_ARXIV,
    _sanitize_query,
    combine_large_chunks,
    fetch_chunks_for_paper,
    fetch_per_query,
)

# ── _sanitize_query ──────────────────────────────────────────────────────────


def test_sanitize_query_removes_hyphen():
    """FTS5 の NOT 演算子になる '-' を除去する。"""
    assert _sanitize_query("key-value cache") == "key value cache"


def test_sanitize_query_removes_special_chars():
    result = _sanitize_query("query: 'hello'")
    assert ":" not in result and "'" not in result
    assert "query" in result and "hello" in result


def test_sanitize_query_strips_whitespace():
    assert _sanitize_query("  hello world  ") == "hello world"


def test_sanitize_query_preserves_alphanumeric():
    assert _sanitize_query("KV cache 2024") == "KV cache 2024"


def test_sanitize_query_empty():
    assert _sanitize_query("") == ""


def test_sanitize_query_japanese_ok():
    """日本語の単語はそのまま通す（FTS5 特殊文字なし）。"""
    result = _sanitize_query("検索クエリ")
    assert "検索クエリ" in result


# ── combine_large_chunks ─────────────────────────────────────────────────────


def _make_chunk(idx: int, text: str, score: float = 0.5) -> dict:
    return {"chunk_index": idx, "text": text, "score": score}


def test_combine_merges_within_target():
    """合計トークンが target 以内なら 1 チャンクに結合する。"""
    chunks = [
        _make_chunk(0, "one two three"),  # 3 words
        _make_chunk(1, "four five six"),  # 3 words
    ]
    result = combine_large_chunks(chunks, paper_id=1, target_tokens=10)
    assert len(result) == 1
    assert "one two three" in result[0]["text"]
    assert "four five six" in result[0]["text"]
    assert result[0]["chunk_indices"] == [0, 1]


def test_combine_splits_on_overflow():
    """合計が target を超えたら新しいチャンクを開始する。"""
    chunks = [
        _make_chunk(0, "one two three four five"),  # 5 words
        _make_chunk(1, "six seven eight nine ten"),  # 5 words
    ]
    result = combine_large_chunks(chunks, paper_id=1, target_tokens=6)
    assert len(result) == 2
    assert result[0]["chunk_indices"] == [0]
    assert result[1]["chunk_indices"] == [1]


def test_combine_takes_max_score():
    """スコアは結合チャンク中の最大値を採用する。"""
    chunks = [
        _make_chunk(0, "short", score=0.3),
        _make_chunk(1, "text", score=0.9),
    ]
    result = combine_large_chunks(chunks, paper_id=1, target_tokens=100)
    assert result[0]["score"] == 0.9


def test_combine_adds_arxiv_id_if_known():
    """PAPER_ID_TO_ARXIV に存在する paper_id なら arxiv_id が付与される。"""
    pid = 1  # PAPER_ID_TO_ARXIV[1] == "2410.10071"
    chunks = [_make_chunk(0, "hello world")]
    result = combine_large_chunks(chunks, paper_id=pid, target_tokens=100)
    assert result[0].get("arxiv_id") == PAPER_ID_TO_ARXIV[pid]


def test_combine_no_arxiv_id_if_unknown_paper():
    """マッピングにない paper_id では arxiv_id フィールドが付与されない。"""
    chunks = [_make_chunk(0, "hello world")]
    result = combine_large_chunks(chunks, paper_id=9999, target_tokens=100)
    assert "arxiv_id" not in result[0]


def test_combine_empty_input():
    assert combine_large_chunks([], paper_id=1, target_tokens=512) == []


def test_combine_preserves_paper_id():
    chunks = [_make_chunk(0, "text")]
    result = combine_large_chunks(chunks, paper_id=42, target_tokens=100)
    assert result[0]["paper_id"] == 42


# ── fetch_chunks_for_paper (mocked) ─────────────────────────────────────────


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_returns_chunks(mock_open):
    mock_open.return_value = MagicMock(
        read=lambda: json.dumps(
            {
                "results": [
                    {"chunk_index": 0, "snippet": "text A", "score": 0.8},
                    {"chunk_index": 1, "snippet": "text B", "score": 0.7},
                ]
            }
        ).encode()
    )
    chunks = fetch_chunks_for_paper("http://api", paper_id=1, query="test")
    assert len(chunks) == 2
    assert chunks[0]["text"] == "text A"
    assert chunks[0]["paper_id"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["score"] == 0.8


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_adds_arxiv_id_for_known_paper(mock_open):
    mock_open.return_value = MagicMock(
        read=lambda: json.dumps(
            {
                "results": [
                    {"chunk_index": 0, "snippet": "x", "score": 0.5},
                ]
            }
        ).encode()
    )
    pid = 1
    chunks = fetch_chunks_for_paper("http://api", paper_id=pid, query="q")
    assert chunks[0]["arxiv_id"] == PAPER_ID_TO_ARXIV[pid]


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_no_arxiv_id_for_unknown_paper(mock_open):
    mock_open.return_value = MagicMock(
        read=lambda: json.dumps(
            {
                "results": [
                    {"chunk_index": 0, "snippet": "x", "score": 0.5},
                ]
            }
        ).encode()
    )
    chunks = fetch_chunks_for_paper("http://api", paper_id=9999, query="q")
    assert "arxiv_id" not in chunks[0]


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_sanitizes_query(mock_open):
    """特殊文字を含むクエリが URL に渡る前にサニタイズされることを確認。"""
    mock_open.return_value = MagicMock(read=lambda: json.dumps({"results": []}).encode())
    fetch_chunks_for_paper("http://api", paper_id=1, query="key-value: 'test'")
    call_url = mock_open.call_args[0][0]
    assert "-" not in call_url
    assert "'" not in call_url


# ── fetch_per_query (mocked fetch_chunks_for_paper) ──────────────────────────


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_deduplicates(mock_fetch):
    """同一チャンクが複数クエリで取得された場合、スコアが高い方が残る。"""
    chunk_low = {"paper_id": 1, "chunk_index": 0, "text": "t", "score": 0.3}
    chunk_high = {"paper_id": 1, "chunk_index": 0, "text": "t", "score": 0.9}
    mock_fetch.side_effect = [[chunk_low], [chunk_high]]

    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["t"]},
        {"paper_id": 1, "query": "q2", "answer_spans": ["t"]},
    ]
    result = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert len(result) == 1
    assert result[0]["score"] == 0.9


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_resolves_arxiv_id(mock_fetch):
    """gold に arxiv_id がある場合、ARXIV_TO_PAPER_ID で paper_id を解決する。"""
    mock_fetch.return_value = [{"paper_id": 1, "chunk_index": 0, "text": "t", "score": 0.5}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["t"]}]

    fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    called_pid = mock_fetch.call_args[0][1]
    assert called_pid == ARXIV_TO_PAPER_ID["2410.10071"]


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_skips_unknown_arxiv_id(mock_fetch, capsys):
    """知らない arxiv_id の gold アイテムはスキップされ警告を出す。"""
    gold = [{"arxiv_id": "9999.99999", "query": "q", "answer_spans": ["t"]}]
    result = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert result == []
    mock_fetch.assert_not_called()
    assert "WARNING" in capsys.readouterr().err


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_returns_all_unique_chunks(mock_fetch):
    """複数論文の gold をまとめると全チャンクが返る。"""
    mock_fetch.side_effect = [
        [{"paper_id": 1, "chunk_index": 0, "text": "paper 1 chunk", "score": 0.8}],
        [{"paper_id": 2, "chunk_index": 0, "text": "paper 2 chunk", "score": 0.7}],
    ]
    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["paper 1"]},
        {"paper_id": 2, "query": "q2", "answer_spans": ["paper 2"]},
    ]
    result = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert len(result) == 2
