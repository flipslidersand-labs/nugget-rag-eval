"""Tests for scripts/fetch_papers.py.

Covers: _sanitize_query, combine_large_chunks,
        fetch_chunks_for_paper (mocked), fetch_per_query (mocked),
        FetchError partial-failure behaviour, retry/backoff behaviour.
"""

from __future__ import annotations

import http.client
import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from nugget_rag.http_util import validate_url
from nugget_rag.paper_registry import PAPER_ID_TO_ARXIV
from scripts.fetch_papers import (
    _MAX_RETRIES,
    ARXIV_TO_PAPER_ID,
    FetchError,
    _sanitize_query,
    _urlopen_with_retry,
    combine_large_chunks,
    fetch_chunks_for_paper,
    fetch_papers,
    fetch_per_query,
)
from scripts.fetch_papers import main as fetch_papers_main


def _http_error(code: int) -> HTTPError:
    return HTTPError(url="http://api", code=code, msg="err", hdrs=None, fp=None)


def _cm(data: bytes) -> MagicMock:
    """Return a urlopen callable mock acting as context manager with .read() -> data."""
    resp = MagicMock()
    resp.read.return_value = data
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=cm)


# ── validate_url (SSRF 対策) ────────────────────────────────────────────────


def test_fetch_validate_url_allows_http():
    validate_url("http://localhost:8020")  # should not raise


def test_fetch_validate_url_allows_https():
    validate_url("https://api.example.com")  # should not raise


def test_fetch_validate_url_rejects_file_scheme():
    with pytest.raises(ValueError, match="file"):
        validate_url("file:///etc/passwd")


def test_fetch_validate_url_rejects_ftp_scheme():
    with pytest.raises(ValueError, match="ftp"):
        validate_url("ftp://internal/resource")


def test_fetch_validate_url_rejects_empty_netloc():
    with pytest.raises(ValueError, match="no host"):
        validate_url("http:///path")


def test_fetch_papers_302_redirect_raises_fetch_error():
    """302 応答は追従せず FetchError（#147: ヘッダ転送防止）。"""
    from tests.conftest import redirect_server

    with redirect_server() as (base_url, hits):
        with pytest.raises(FetchError, match="302"):
            fetch_papers(base_url)
    assert len(hits) == 1


@patch("scripts.fetch_papers.urlopen")
def test_fetch_papers_rejects_file_scheme(mock_open):
    with pytest.raises(ValueError, match="file"):
        fetch_papers("file:///etc/passwd")
    mock_open.assert_not_called()


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_rejects_file_scheme(mock_open):
    with pytest.raises(ValueError, match="file"):
        fetch_chunks_for_paper("file:///etc/passwd", paper_id=1, query="q")
    mock_open.assert_not_called()


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_rejects_ftp_scheme(mock_open):
    with pytest.raises(ValueError, match="ftp"):
        fetch_chunks_for_paper("ftp://internal/api", paper_id=1, query="q")
    mock_open.assert_not_called()


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


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize(
    ("sizes", "target", "expected_indices"),
    [
        # (1) 合計がちょうど target なら結合する(<= の等号境界)
        ([3, 3], 6, [[0, 1]]),
        # (2a) 先頭が単独で target 超過 → それ単独で 1 チャンク、次は新しいチャンクを始める
        ([8, 2], 6, [[0], [1]]),
        # (2b) 途中が単独で target 超過 → 前後と結合せず単独で出る
        ([2, 8, 2], 6, [[0], [1], [2]]),
        # (3) 0 語(空 text)は size 0 として結合される。ちょうど target の直後でも入る
        ([6, 0], 6, [[0, 1]]),
        ([0, 3, 0, 3], 6, [[0, 1, 2, 3]]),
        # (4) 3 チャンク以上の連鎖で current_size が累積する(2+2+2=6 は結合、次の 1 は溢れる)
        ([2, 2, 2, 1], 6, [[0, 1, 2], [3]]),
    ],
)
def test_combine_size_boundaries(sizes, target, expected_indices):
    """結合可否は current_size + chunk_size <= target_tokens。超過チャンクは分割せず単独で出す。"""
    chunks = [_make_chunk(i, _words(n)) for i, n in enumerate(sizes)]
    result = combine_large_chunks(chunks, paper_id=1, target_tokens=target)
    assert [r["chunk_indices"] for r in result] == expected_indices
    # 超過チャンクの text は分割・切り詰めされずそのまま出る
    for r in result:
        if len(r["chunk_indices"]) == 1:
            assert r["text"] == chunks[r["chunk_indices"][0]]["text"]


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


def test_combine_sorts_by_chunk_index_before_merging():
    """スコア順で渡されても chunk_index 昇順(文書順)に整列して結合する (#247)。"""
    chunks = [
        _make_chunk(3, "three", score=0.9),
        _make_chunk(40, "forty", score=0.8),
        _make_chunk(4, "four", score=0.1),
    ]
    snapshot = [dict(c) for c in chunks]
    result = combine_large_chunks(chunks, paper_id=1, target_tokens=100)
    assert len(result) == 1
    assert result[0]["chunk_indices"] == [3, 4, 40]
    assert result[0]["text"] == "three four forty"
    assert result[0]["score"] == 0.9
    assert chunks == snapshot  # 入力は破壊しない


def test_combine_sorts_before_overflow_split():
    """整列は結合前に行うので、オーバーフローの切れ目も文書順で決まる (#247)。"""
    chunks = [_make_chunk(2, "c c c"), _make_chunk(0, "a a a"), _make_chunk(1, "b b b")]
    result = combine_large_chunks(chunks, paper_id=1, target_tokens=6)
    assert [r["chunk_indices"] for r in result] == [[0, 1], [2]]
    assert result[0]["text"] == "a a a b b b"


def test_combine_empty_input():
    assert combine_large_chunks([], paper_id=1, target_tokens=512) == []


def test_combine_preserves_paper_id():
    chunks = [_make_chunk(0, "text")]
    result = combine_large_chunks(chunks, paper_id=42, target_tokens=100)
    assert result[0]["paper_id"] == 42


# ── fetch_chunks_for_paper (mocked) ─────────────────────────────────────────


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_returns_chunks(mock_open):
    mock_open.return_value = _cm(
        json.dumps(
            {
                "results": [
                    {"chunk_index": 0, "snippet": "text A", "score": 0.8},
                    {"chunk_index": 1, "snippet": "text B", "score": 0.7},
                ]
            }
        ).encode()
    ).return_value
    chunks = fetch_chunks_for_paper("http://api", paper_id=1, query="test")
    assert len(chunks) == 2
    assert chunks[0]["text"] == "text A"
    assert chunks[0]["paper_id"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["score"] == 0.8


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_adds_arxiv_id_for_known_paper(mock_open):
    mock_open.return_value = _cm(
        json.dumps({"results": [{"chunk_index": 0, "snippet": "x", "score": 0.5}]}).encode()
    ).return_value
    pid = 1
    chunks = fetch_chunks_for_paper("http://api", paper_id=pid, query="q")
    assert chunks[0]["arxiv_id"] == PAPER_ID_TO_ARXIV[pid]


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_no_arxiv_id_for_unknown_paper(mock_open):
    mock_open.return_value = _cm(
        json.dumps({"results": [{"chunk_index": 0, "snippet": "x", "score": 0.5}]}).encode()
    ).return_value
    chunks = fetch_chunks_for_paper("http://api", paper_id=9999, query="q")
    assert "arxiv_id" not in chunks[0]


@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_sanitizes_query(mock_open):
    """特殊文字を含むクエリが URL に渡る前にサニタイズされることを確認。"""
    mock_open.return_value = _cm(json.dumps({"results": []}).encode()).return_value
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
    result, failures = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert len(result) == 1
    assert result[0]["score"] == 0.9
    assert failures == 0


def _raw(pid: int, idx: int, score: float = 0.5) -> dict:
    return {"paper_id": pid, "chunk_index": idx, "text": f"body{idx}", "score": score}


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_large_shared_chunk_appears_once(mock_fetch):
    """large: クエリ1=[3,5]・クエリ2=[3,7] でも chunk 3 は 1 回だけ出る (#266)。"""
    mock_fetch.side_effect = [[_raw(1, 3), _raw(1, 5)], [_raw(1, 3), _raw(1, 7)]]
    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["x"]},
        {"paper_id": 1, "query": "q2", "answer_spans": ["x"]},
    ]
    result, failures = fetch_per_query("http://api", gold, chunk_mode="large", target_tokens=512)
    assert failures == 0
    all_indices = [i for c in result for i in c["chunk_indices"]]
    assert sorted(all_indices) == [3, 5, 7]
    assert sum(c["text"].split().count("body3") for c in result) == 1
    # 固定: 重複排除後の和集合を 1 回で結合する (文書順 3,5,7)。
    assert len(result) == 1
    assert result[0]["chunk_indices"] == [3, 5, 7]
    assert result[0]["text"] == "body3 body5 body7"


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_large_orders_by_chunk_index_across_queries(mock_fetch):
    """large: クエリ1=[3,40]・クエリ2=[4] でも文書順 3,4,40 で結合する (#247)。"""
    mock_fetch.side_effect = [[_raw(1, 3), _raw(1, 40)], [_raw(1, 4)]]
    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["x"]},
        {"paper_id": 1, "query": "q2", "answer_spans": ["x"]},
    ]
    result, _ = fetch_per_query("http://api", gold, chunk_mode="large", target_tokens=512)
    assert [c["chunk_indices"] for c in result] == [[3, 4, 40]]
    assert result[0]["text"] == "body3 body4 body40"


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_large_shared_chunk_keeps_highest_score(mock_fetch):
    """large: 重複チャンクは高スコア側が残り、結合チャンクの score に反映される (#266)。"""
    mock_fetch.side_effect = [[_raw(1, 3, 0.2)], [_raw(1, 3, 0.9)]]
    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["x"]},
        {"paper_id": 1, "query": "q2", "answer_spans": ["x"]},
    ]
    result, _ = fetch_per_query("http://api", gold, chunk_mode="large", target_tokens=512)
    assert len(result) == 1
    assert result[0]["chunk_indices"] == [3]
    assert result[0]["score"] == 0.9


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_large_combines_per_paper(mock_fetch):
    """large: 論文ごとに別々に結合し、論文をまたいでは結合しない (#266)。"""
    mock_fetch.side_effect = [[_raw(1, 0)], [_raw(2, 0)], [_raw(1, 1)]]
    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["x"]},
        {"paper_id": 2, "query": "q2", "answer_spans": ["x"]},
        {"paper_id": 1, "query": "q3", "answer_spans": ["x"]},
    ]
    result, _ = fetch_per_query("http://api", gold, chunk_mode="large", target_tokens=512)
    assert [(c["paper_id"], c["chunk_indices"]) for c in result] == [(1, [0, 1]), (2, [0])]


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_small_drops_blank_snippets(mock_fetch):
    """small モード: 空白だけの snippet は出力しない (1件で chunks 全体が不正になる)。"""
    blank = {"paper_id": 1, "chunk_index": 0, "text": "   \n ", "score": 0.9}
    good = {"paper_id": 1, "chunk_index": 1, "text": "real text", "score": 0.5}
    mock_fetch.return_value = [blank, good]

    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["real"]}]
    result, failures = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert [c["text"] for c in result] == ["real text"]
    assert failures == 0


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
    result, failures = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert result == []
    assert failures == 1  # #230: counted so --max-failures can gate it
    mock_fetch.assert_not_called()
    assert "WARNING" in capsys.readouterr().err


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_null_arxiv_id_falls_back_to_paper_id(mock_fetch):
    """arxiv_id が null/空でも有効な paper_id が使われ、失敗に数えない (#230)。"""
    mock_fetch.return_value = [{"paper_id": 7, "chunk_index": 0, "text": "t", "score": 0.5}]
    gold = [
        {"arxiv_id": None, "paper_id": 7, "query": "q", "answer_spans": ["t"]},
        {"arxiv_id": "", "paper_id": 7, "query": "q2", "answer_spans": ["t"]},
    ]
    result, failures = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert failures == 0
    assert [c[0][1] for c in mock_fetch.call_args_list] == [7, 7]
    assert len(result) == 1


def test_main_unregistered_arxiv_id_exits_1_by_default(tmp_path, monkeypatch):
    """未登録 arxiv_id を含む gold は既定 (--max-failures 未指定=閾値0) で exit 1 (#230)。
    未登録はネットワーク前にスキップされるため実通信は発生しない。"""
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps([{"arxiv_id": "9999.99999", "query": "q", "answer_spans": ["t"]}]),
        encoding="utf-8",
    )
    out = tmp_path / "chunks.json"
    monkeypatch.setattr("sys.argv", ["fetch_papers", "--gold-set", str(gold), "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        fetch_papers_main()
    assert exc.value.code == 1


def test_main_empty_result_keeps_existing_out_gold_mode(tmp_path, monkeypatch, capsys):
    """全件スキップで chunks が 0 件なら exit 1 し、既存 out を上書きしない (#249)。"""
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps([{"arxiv_id": "9999.99999", "query": "q", "answer_spans": ["t"]}]),
        encoding="utf-8",
    )
    out = tmp_path / "chunks.json"
    out.write_text('[{"keep": true}]', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["fetch_papers", "--gold-set", str(gold), "--out", str(out), "--max-failures", "10"],
    )
    with pytest.raises(SystemExit) as exc:
        fetch_papers_main()
    assert exc.value.code == 1
    assert "[ERROR]" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == '[{"keep": true}]'
    assert not out.with_suffix(".tmp").exists()


@patch("scripts.fetch_papers.fetch_papers", return_value=[])
def test_main_empty_result_keeps_existing_out_papers_mode(_mock, tmp_path, monkeypatch, capsys):
    """/papers が空でも exit 1 し、既存 out を保持する (#249)。"""
    out = tmp_path / "chunks.json"
    out.write_text('[{"keep": true}]', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["fetch_papers", "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        fetch_papers_main()
    assert exc.value.code == 1
    assert "[ERROR]" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == '[{"keep": true}]'


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
    result, failures = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    assert len(result) == 2
    assert failures == 0


# ── _urlopen_with_retry ──────────────────────────────────────────────────────


# ── 応答サイズ上限 (#305) ────────────────────────────────────────────────────


@patch("scripts.fetch_papers.MAX_RESPONSE_BYTES", 10)
@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_oversized_response_raises_fetch_error_without_retry(mock_open, mock_sleep):
    cm = _cm(b"x" * 11)
    mock_open.return_value = cm.return_value
    with pytest.raises(FetchError, match="exceeds 10 bytes"):
        _urlopen_with_retry("http://api", timeout=5)
    assert mock_open.call_count == 1
    mock_sleep.assert_not_called()
    cm.return_value.__enter__.return_value.read.assert_called_once_with(11)


@patch("scripts.fetch_papers.MAX_RESPONSE_BYTES", 10)
@patch("scripts.fetch_papers.urlopen")
def test_response_at_size_limit_is_accepted(mock_open):
    mock_open.return_value = _cm(b"x" * 10).return_value
    assert _urlopen_with_retry("http://api", timeout=5) == b"x" * 10


@patch("scripts.fetch_papers.MAX_RESPONSE_BYTES", 10)
@patch("scripts.fetch_papers.urlopen")
def test_fetch_papers_oversized_response_is_fetch_error(mock_open):
    mock_open.return_value = _cm(b" " * 11).return_value
    with pytest.raises(FetchError, match="exceeds"):
        fetch_papers("http://localhost:8020")


@patch("scripts.fetch_papers.MAX_RESPONSE_BYTES", 10)
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_oversized_response_is_fetch_error(mock_open):
    mock_open.return_value = _cm(b" " * 11).return_value
    with pytest.raises(FetchError, match="exceeds"):
        fetch_chunks_for_paper("http://localhost:8020", 1, "q")


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_truncated_body_is_retried_when_content_length_not_met(mock_open, mock_sleep):
    """read(amt) does not raise IncompleteRead on a short body; keep retrying (#232)."""
    short = _cm(b"par")
    short.return_value.__enter__.return_value.headers = {"Content-Length": "7"}
    mock_open.side_effect = [short.return_value, _cm(b"payload").return_value]
    assert _urlopen_with_retry("http://api", timeout=5) == b"payload"
    assert mock_open.call_count == 2


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_succeeds_after_transient_urlerror(mock_open, mock_sleep):
    """transient エラー（1〜2 回失敗後成功）でボディが返る＝chunk 欠落なし。"""
    ok = _cm(b"payload").return_value
    mock_open.side_effect = [URLError("reset"), URLError("reset"), ok]
    assert _urlopen_with_retry("http://api", timeout=5) == b"payload"
    assert mock_open.call_count == 3
    assert mock_sleep.call_count == 2


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_succeeds_after_transient_5xx(mock_open, mock_sleep):
    """5xx はリトライ対象。"""
    ok = _cm(b"payload").return_value
    mock_open.side_effect = [_http_error(503), ok]
    assert _urlopen_with_retry("http://api", timeout=5) == b"payload"
    assert mock_open.call_count == 2


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_4xx_raises_immediately(mock_open, mock_sleep):
    """4xx は恒久失敗として即 raise（リトライ・sleep なし）。"""
    mock_open.side_effect = _http_error(404)
    with pytest.raises(HTTPError):
        _urlopen_with_retry("http://api", timeout=5)
    assert mock_open.call_count == 1
    mock_sleep.assert_not_called()


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_exhaustion_raises_last_error(mock_open, mock_sleep):
    """リトライ上限まで失敗したら最後のエラーが raise される。"""
    mock_open.side_effect = URLError("down")
    with pytest.raises(URLError):
        _urlopen_with_retry("http://api", timeout=5, max_retries=2)
    assert mock_open.call_count == 3  # initial + 2 retries
    assert mock_sleep.call_count == 2


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_backoff_is_exponential(mock_open, mock_sleep):
    """バックオフは retry_backoff * 2**attempt で伸びる。"""
    mock_open.side_effect = URLError("down")
    with pytest.raises(URLError):
        _urlopen_with_retry("http://api", timeout=5, max_retries=2, retry_backoff=1.0)
    assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0, 2.0]


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_recovers_from_transient_error(mock_open, mock_sleep):
    """fetch_chunks_for_paper は transient エラー後にリトライで chunks を返す。"""
    ok = _cm(
        json.dumps({"results": [{"chunk_index": 0, "snippet": "x", "score": 0.5}]}).encode()
    ).return_value
    mock_open.side_effect = [URLError("reset"), ok]
    chunks = fetch_chunks_for_paper("http://api", paper_id=1, query="q")
    assert len(chunks) == 1


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_fetch_papers_4xx_raises_fetch_error_without_retry(mock_open, mock_sleep):
    """fetch_papers は 4xx を即 FetchError に変換する。"""
    mock_open.side_effect = _http_error(400)
    with pytest.raises(FetchError, match="Failed to fetch papers"):
        fetch_papers("http://api")
    assert mock_open.call_count == 1
    mock_sleep.assert_not_called()


# ── FetchError / partial-failure behaviour ───────────────────────────────────


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_raises_fetch_error_on_network_failure(mock_open, mock_sleep):
    """URLError が（リトライ枯渇後）FetchError に変換されて raise される（sys.exit しない）。"""
    mock_open.side_effect = URLError("connection refused")
    with pytest.raises(FetchError, match="Failed to fetch chunks for paper"):
        fetch_chunks_for_paper("http://api", paper_id=1, query="test")


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_skips_failed_item_by_default(mock_fetch, capsys):
    """fetch_chunks_for_paper が FetchError を上げても、fail_fast=False ならスキップして続行。"""
    mock_fetch.side_effect = [
        FetchError("network error"),
        [{"paper_id": 2, "chunk_index": 0, "text": "ok", "score": 0.7}],
    ]
    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["x"]},
        {"paper_id": 2, "query": "q2", "answer_spans": ["ok"]},
    ]
    result, failures = fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512)
    # second item must be preserved despite first failure
    assert len(result) == 1
    assert result[0]["paper_id"] == 2
    assert failures == 1
    assert "WARNING" in capsys.readouterr().err


@patch("scripts.fetch_papers.fetch_chunks_for_paper")
def test_fetch_per_query_fail_fast_raises(mock_fetch):
    """fail_fast=True のとき FetchError が即座に伝播する。"""
    mock_fetch.side_effect = FetchError("boom")
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["x"]}]
    with pytest.raises(FetchError, match="boom"):
        fetch_per_query("http://api", gold, chunk_mode="small", target_tokens=512, fail_fast=True)


_BAD_RESULT_ITEMS = [
    {"chunk_index": 0, "snippet": None, "score": 0.5},
    {"chunk_index": [1, 2], "snippet": "t", "score": 0.5},
    {"chunk_index": True, "snippet": "t", "score": 0.5},
    {"chunk_index": 0, "snippet": "t", "score": "high"},
]


@pytest.mark.parametrize("item", _BAD_RESULT_ITEMS)
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_wrong_value_types_raise_fetch_error(mock_open, item):
    """値の型が想定外 (null snippet / list chunk_index 等) でも FetchError (#241)。"""
    mock_open.return_value = _cm(json.dumps({"results": [item]}).encode()).return_value
    with pytest.raises(FetchError, match="Malformed search response"):
        fetch_chunks_for_paper("http://api", paper_id=1, query="test")


@pytest.mark.parametrize("chunk_mode", ["small", "large"])
@patch("scripts.fetch_papers.urlopen")
def test_fetch_per_query_wrong_value_types_counted_and_continues(mock_open, chunk_mode, capsys):
    """型不正の応答は failure_count に加算され、次の項目の処理が継続する (#241)。"""
    bad = {"results": [{"chunk_index": [1], "snippet": None, "score": 0.5}]}
    good = {"results": [{"chunk_index": 0, "snippet": "ok text", "score": 0.7}]}
    mock_open.side_effect = [
        _cm(json.dumps(bad).encode()).return_value,
        _cm(json.dumps(good).encode()).return_value,
    ]
    gold = [
        {"paper_id": 1, "query": "q1", "answer_spans": ["x"]},
        {"paper_id": 2, "query": "q2", "answer_spans": ["ok"]},
    ]
    result, failures = fetch_per_query("http://api", gold, chunk_mode=chunk_mode, target_tokens=512)
    assert failures == 1
    assert len(result) == 1
    assert result[0]["paper_id"] == 2
    assert "WARNING" in capsys.readouterr().err


# ── main() subprocess tests ──────────────────────────────────────────────────


import subprocess  # noqa: E402
import sys as _sys  # noqa: E402


def _run_main(*args: str) -> subprocess.CompletedProcess:
    """Run scripts/fetch_papers.py main() in a subprocess and return the result."""
    return subprocess.run(
        [_sys.executable, "-m", "scripts.fetch_papers", *args],
        capture_output=True,
        text=True,
    )


def test_main_missing_gold_set_prints_error_and_exits_1(tmp_path):
    """`--gold-set /nonexistent` → [ERROR] on stderr, exit 1 (no traceback)."""
    result = _run_main("--gold-set", str(tmp_path / "nonexistent.json"))
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "Traceback" not in result.stderr


def test_main_invalid_json_gold_set_prints_error_and_exits_1(tmp_path):
    """壊れた JSON の gold-set → [ERROR] on stderr, exit 1 (no traceback)."""
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    result = _run_main("--gold-set", str(bad))
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "Traceback" not in result.stderr


def test_main_invalid_gold_set_schema_prints_error_and_exits_1(tmp_path):
    """schema 違反の gold-set (query フィールド欠落) → [ERROR] on stderr, exit 1."""
    bad = tmp_path / "gold.json"
    bad.write_text(
        '[{"paper_id": 1, "answer_spans": ["span"]}]',  # missing 'query'
        encoding="utf-8",
    )
    result = _run_main("--gold-set", str(bad))
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "Traceback" not in result.stderr


def test_main_bare_api_url_prints_error_and_exits_2():
    """スキーム省略の --api-url (localhost:8020) → argparse error, exit 2."""
    result = _run_main("--api-url", "localhost:8020", "--gold-set", "/dummy")
    assert result.returncode == 2
    assert "did you mean" in result.stderr.lower() or "http://" in result.stderr


def test_main_invalid_url_scheme_exits_2():
    """非 http(s) スキームの --api-url → argparse error, exit 2."""
    result = _run_main("--api-url", "ftp://localhost:8020", "--gold-set", "/dummy")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_main_negative_max_failures_exits_2():
    """--max-failures に負値 → argparse error, exit 2."""
    result = _run_main("--max-failures", "-1")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_main_max_failures_zero_is_valid(tmp_path):
    """--max-failures 0 は有効値（境界値）→ argparse エラーにならない。
    gold-set 欠落の [ERROR] が先に出るので exit 1 になる。"""
    result = _run_main("--max-failures", "0", "--gold-set", str(tmp_path / "nope.json"))
    # --max-failures=0 itself is accepted; the gold-set error causes exit 1
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "Traceback" not in result.stderr


# ── 200 応答の本文不正は FetchError (#234) ───────────────────────────────────

_BAD_CHUNK_BODIES = [
    b"not json",
    b"{}",
    b'{"results": null}',
    b'{"results": [{"chunk_index": 0}]}',
    b'{"results": [1]}',
]

_BAD_PAPERS_BODIES = [
    b"not json",
    b"{}",
    b'{"papers": null}',
    b'{"papers": [{"name": "x"}]}',
    b'{"papers": [1]}',
]


@pytest.mark.parametrize("body", _BAD_CHUNK_BODIES)
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_malformed_body_raises_fetch_error(mock_open, body):
    mock_open.return_value = _cm(body).return_value
    with pytest.raises(FetchError):
        fetch_chunks_for_paper("http://api", paper_id=1, query="test")


@pytest.mark.parametrize("body", _BAD_PAPERS_BODIES)
@patch("scripts.fetch_papers.urlopen")
def test_fetch_papers_malformed_body_raises_fetch_error(mock_open, body):
    mock_open.return_value = _cm(body).return_value
    with pytest.raises(FetchError):
        fetch_papers("http://api")


@patch("scripts.fetch_papers.urlopen")
def test_fetch_per_query_counts_malformed_body_and_continues(mock_open):
    good = json.dumps({"results": [{"chunk_index": 0, "snippet": "t", "score": 0.5}]}).encode()
    mock_open.side_effect = [
        _cm(b"not json").return_value,
        _cm(b'{"results": null}').return_value,
        _cm(good).return_value,
    ]
    gold = [{"paper_id": 1, "query": f"q{i}"} for i in range(3)]
    chunks, failures = fetch_per_query("http://api", gold, "small", 512)
    assert failures == 2
    assert len(chunks) == 1


@pytest.mark.parametrize("body", [b'{"papers": [{"id": null}]}', b'{"papers": [{}]}'])
@patch("scripts.fetch_papers.urlopen")
def test_fetch_papers_null_or_missing_id_raises_fetch_error(mock_open, body):
    mock_open.return_value = _cm(body).return_value
    with pytest.raises(FetchError):
        fetch_papers("http://api")


@patch("scripts.fetch_papers.urlopen")
def test_fetch_papers_list_id_raises_fetch_error(mock_open):
    mock_open.return_value = _cm(b'{"papers": [{"id": [1]}]}').return_value
    with pytest.raises(FetchError):
        fetch_papers("http://api")


@pytest.mark.parametrize("body", [b'{"results": {}}', b'{"results": "x"}'])
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_non_list_results_raises_fetch_error(mock_open, body):
    mock_open.return_value = _cm(body).return_value
    with pytest.raises(FetchError):
        fetch_chunks_for_paper("http://api", paper_id=1, query="test")


# ── read() 中のタイムアウト/切断 (#265) ───────────────────────────────────────


def _read_raises_cm(exc: BaseException) -> MagicMock:
    """urlopen() は成功するが read() が exc を投げるレスポンスのモック。"""
    resp = MagicMock()
    resp.read.side_effect = exc
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


_READ_ERRORS = [
    pytest.param(lambda: TimeoutError("timed out"), id="timeout"),
    pytest.param(lambda: ConnectionResetError("reset"), id="conn-reset"),
    pytest.param(lambda: http.client.IncompleteRead(b"par", 10), id="incomplete-read"),
]


@pytest.mark.parametrize("make_exc", _READ_ERRORS)
@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_succeeds_after_read_error(mock_open, mock_sleep, make_exc):
    """read() 中の例外は再試行され、次の試行で成功すればボディが返る。"""
    ok = _cm(b"payload").return_value
    mock_open.side_effect = [_read_raises_cm(make_exc()), ok]
    assert _urlopen_with_retry("http://api", timeout=5) == b"payload"
    assert mock_open.call_count == 2
    assert mock_sleep.call_count == 1


@pytest.mark.parametrize("make_exc", _READ_ERRORS)
@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_read_error_exhaustion_raises_fetch_error(mock_open, mock_sleep, make_exc):
    """再試行を使い切ったら FetchError（traceback で止まらない）。"""
    mock_open.side_effect = lambda *a, **k: _read_raises_cm(make_exc())
    with pytest.raises(FetchError, match="Failed to fetch chunks for paper 1"):
        fetch_chunks_for_paper("http://api", paper_id=1, query="q")
    assert mock_open.call_count == _MAX_RETRIES + 1
    assert mock_sleep.call_count == _MAX_RETRIES


@pytest.mark.parametrize("make_exc", _READ_ERRORS)
@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_fetch_papers_read_error_exhaustion_raises_fetch_error(mock_open, mock_sleep, make_exc):
    mock_open.side_effect = lambda *a, **k: _read_raises_cm(make_exc())
    with pytest.raises(FetchError, match="Failed to fetch papers"):
        fetch_papers("http://api")
    assert mock_open.call_count == _MAX_RETRIES + 1


# ── 408/429 の再試行と Retry-After (#284) ────────────────────────────────────


def _http_error_ra(code: int, retry_after: str | None) -> HTTPError:
    from email.message import Message

    hdrs = Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return HTTPError(url="http://api", code=code, msg="err", hdrs=hdrs, fp=None)


@pytest.mark.parametrize("code", [408, 429])
@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_408_429_then_success(mock_open, mock_sleep, code):
    ok = _cm(b"payload").return_value
    mock_open.side_effect = [_http_error(code), ok]
    assert _urlopen_with_retry("http://api", timeout=5) == b"payload"
    assert mock_open.call_count == 2


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7),
        ("999", 30),
        ("abc", 1.0),
        ("-3", 1.0),
        ("1.5", 1.0),
        (None, 1.0),
        ("9" * 5000, 1.0),
    ],
)
@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_after_is_capped_and_invalid_ignored(mock_open, mock_sleep, header, expected):
    ok = _cm(b"payload").return_value
    mock_open.side_effect = [_http_error_ra(429, header), ok]
    _urlopen_with_retry("http://api", timeout=5)
    assert [c.args[0] for c in mock_sleep.call_args_list] == [expected]


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_retry_5xx_closes_http_error_before_retry_and_on_exhaustion(mock_open, mock_sleep):
    """5xx の HTTPError は再試行前・最終 raise 前に close される（FD リーク防止, #280）。"""
    errors = [_http_error(503) for _ in range(3)]
    for e in errors:
        e.close = MagicMock()
    mock_open.side_effect = errors
    with pytest.raises(HTTPError):
        _urlopen_with_retry("http://api", timeout=5, max_retries=2)
    assert all(e.close.call_count >= 1 for e in errors)


@patch("nugget_rag.http_util.time.sleep")
@patch("scripts.fetch_papers.urlopen")
def test_non_retryable_4xx_closes_http_error_before_raising(mock_open, mock_sleep):
    """再試行しない 4xx の HTTPError も close してから raise する（FD リーク防止, #313）。"""
    err = _http_error(404)
    err.close = MagicMock()
    mock_open.side_effect = err
    with pytest.raises(HTTPError):
        _urlopen_with_retry("http://api", timeout=5, max_retries=2)
    assert mock_open.call_count == 1
    err.close.assert_called_once()
    mock_sleep.assert_not_called()


_SMALL_KEYS = ["paper_id", "chunk_index", "text", "score", "arxiv_id"]
_LARGE_KEYS = ["paper_id", "chunk_indices", "text", "score", "arxiv_id"]


@pytest.mark.parametrize(
    ("pid", "drop_arxiv"), [(1, False), (9999, True)], ids=["known", "unknown"]
)
@patch("scripts.fetch_papers.urlopen")
def test_fetch_chunks_key_order(mock_open, pid, drop_arxiv):
    """small モードのキー順は出力 JSON の一部。arxiv_id は末尾 (#336)。"""
    mock_open.return_value = _cm(
        json.dumps({"results": [{"chunk_index": 0, "snippet": "x", "score": 0.5}]}).encode()
    ).return_value
    chunks = fetch_chunks_for_paper("http://api", paper_id=pid, query="q")
    assert list(chunks[0]) == (_SMALL_KEYS[:-1] if drop_arxiv else _SMALL_KEYS)


@pytest.mark.parametrize(
    ("pid", "drop_arxiv"), [(1, False), (9999, True)], ids=["known", "unknown"]
)
def test_combine_key_order_first_and_after_split(pid, drop_arxiv):
    """large モードのキー順: 先頭チャンクと区切り後のチャンクの両方で同一 (#336)。"""
    chunks = [_make_chunk(0, "a b c"), _make_chunk(1, "d e f"), _make_chunk(2, "g h i")]
    result = combine_large_chunks(chunks, paper_id=pid, target_tokens=4)
    assert len(result) == 3
    expected = _LARGE_KEYS[:-1] if drop_arxiv else _LARGE_KEYS
    for chunk in result:
        assert list(chunk) == expected
