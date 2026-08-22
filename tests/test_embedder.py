import json as _json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from nugget_rag.embedder import MAX_BATCH_SIZE, EmbedClient, cosine_similarity, embed_scores
from nugget_rag.scorer import top_nuggets

# --- EmbedClient.embed() エラーパス ---


def _client():
    return EmbedClient("http://localhost:9092", api_key="test-key")


def test_embed_empty_texts_returns_empty():
    assert _client().embed([]) == []


def test_embed_url_error_raises():
    with patch("nugget_rag.embedder.urlopen", side_effect=URLError("refused")):
        with pytest.raises((URLError, Exception)):
            _client().embed(["hello"])


def test_embed_http_error_raises():
    err = HTTPError("http://x", 500, "Server Error", {}, None)
    with patch("nugget_rag.embedder.urlopen", side_effect=err):
        with pytest.raises(Exception):
            _client().embed(["hello"])


def test_embed_malformed_response_missing_vectors_key_raises():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"embeddings": [[1.0, 2.0]]}'
    with patch("nugget_rag.embedder.urlopen", return_value=mock_resp):
        with pytest.raises((KeyError, Exception)):
            _client().embed(["hello"])


def test_embed_invalid_json_raises():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"not json"
    with patch("nugget_rag.embedder.urlopen", return_value=mock_resp):
        with pytest.raises(Exception):
            _client().embed(["hello"])


def test_embed_success_returns_vectors():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"vectors": [[0.1, 0.2], [0.3, 0.4]]}'
    with patch("nugget_rag.embedder.urlopen", return_value=mock_resp):
        result = _client().embed(["text1", "text2"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == 1.0


def test_cosine_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_embed_scores_length():
    query_vec = [1.0, 0.0]
    sent_vecs = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]
    scores = embed_scores(query_vec, sent_vecs)
    assert len(scores) == 3


def test_embed_scores_ranking():
    query_vec = [1.0, 0.0]
    sent_vecs = [[0.0, 1.0], [1.0, 0.0]]  # second is identical to query
    scores = embed_scores(query_vec, sent_vecs)
    assert scores[1] > scores[0]


def test_top_nuggets_with_embed_fn():
    sentences = [
        "KV cache reuse reduces inference cost.",
        "Diffusion models generate images.",
        "KV caching is essential for transformer efficiency.",
    ]

    # Mock embed_fn: returns fixed vectors where KV sentences are close to query
    def mock_embed(texts):
        vecs = {
            "KV cache reuse": [1.0, 0.0, 0.0],
            "KV cache reuse reduces inference cost.": [0.9, 0.1, 0.0],
            "Diffusion models generate images.": [0.0, 1.0, 0.0],
            "KV caching is essential for transformer efficiency.": [0.8, 0.1, 0.1],
        }
        return [vecs.get(t, [0.0, 0.0, 1.0]) for t in texts]

    result = top_nuggets(
        "KV cache reuse", sentences, top_k=2, embed_fn=mock_embed, embed_weight=0.8
    )
    assert len(result) == 2
    # Both KV-related sentences should win
    assert all("KV" in r or "kv" in r.lower() for r in result)


def test_top_nuggets_embed_fn_none_unchanged():
    """Without embed_fn, behavior is identical to BM25-only."""
    sentences = ["KV cache method", "Diffusion images"]
    result_no_embed = top_nuggets("KV cache", sentences, top_k=1)
    result_bm25 = top_nuggets("KV cache", sentences, top_k=1, embed_fn=None)
    assert result_no_embed == result_bm25


# --- バッチ分割テスト (MAX_BATCH_SIZE) ---



@patch("nugget_rag.embedder.urlopen")
def test_embed_splits_large_batch(mock_open):
    """texts > MAX_BATCH_SIZE のとき urlopen が複数回呼ばれる。"""

    def side_effect(req, timeout):
        batch_size = len(_json.loads(req.data)["texts"])
        resp = MagicMock()
        resp.read.return_value = _json.dumps({"vectors": [[0.1, 0.2]] * batch_size}).encode()
        return resp

    mock_open.side_effect = side_effect

    n = MAX_BATCH_SIZE + 10
    vecs = _client().embed(["text"] * n)
    assert len(vecs) == n
    assert mock_open.call_count == 2  # 256 + 10


@patch("nugget_rag.embedder.urlopen")
def test_embed_exact_batch_size_single_call(mock_open):
    """texts == MAX_BATCH_SIZE のとき urlopen は 1 回だけ。"""
    mock_open.return_value = MagicMock(
        read=lambda: __import__("json").dumps({"vectors": [[0.0]] * MAX_BATCH_SIZE}).encode()
    )
    vecs = _client().embed(["t"] * MAX_BATCH_SIZE)
    assert len(vecs) == MAX_BATCH_SIZE
    assert mock_open.call_count == 1


@patch("nugget_rag.embedder.urlopen")
def test_embed_batch_preserves_order(mock_open):
    """複数バッチでもベクトルの順序が保たれる。"""
    call_idx = [0]

    def side_effect(req, timeout):
        batch = _json.loads(req.data)["texts"]
        vecs = [[float(i + call_idx[0] * MAX_BATCH_SIZE)] for i in range(len(batch))]
        call_idx[0] += 1
        resp = MagicMock()
        resp.read.return_value = _json.dumps({"vectors": vecs}).encode()
        return resp

    mock_open.side_effect = side_effect
    n = MAX_BATCH_SIZE + 5
    vecs = _client().embed(["t"] * n)
    assert len(vecs) == n
    for i, v in enumerate(vecs):
        assert v == [float(i)]


# --- リトライテスト (max_retries) ---


def _retry_client(max_retries=3, backoff=0.0):
    return EmbedClient(
        "http://localhost:9092", api_key="k", max_retries=max_retries, retry_backoff=backoff
    )


def _ok_response(n=1):
    m = MagicMock()
    m.read.return_value = _json.dumps({"vectors": [[0.1]] * n}).encode()
    return m


@patch("nugget_rag.embedder.time.sleep")
@patch("nugget_rag.embedder.urlopen")
def test_retry_succeeds_on_second_attempt(mock_open, mock_sleep):
    """1 回目 URLError → 2 回目成功 → ベクトルが返る。"""
    mock_open.side_effect = [URLError("transient"), _ok_response()]
    vecs = _retry_client(max_retries=3).embed(["t"])
    assert vecs == [[0.1]]
    assert mock_open.call_count == 2


@patch("nugget_rag.embedder.time.sleep")
@patch("nugget_rag.embedder.urlopen")
def test_retry_exhausted_raises_embed_error(mock_open, mock_sleep):
    """max_retries 回すべて URLError → EmbedError。"""
    from nugget_rag.embedder import EmbedError

    mock_open.side_effect = URLError("down")
    with pytest.raises(EmbedError, match="retries"):
        _retry_client(max_retries=2, backoff=0.0).embed(["t"])
    assert mock_open.call_count == 3  # 初回 + 2 回リトライ


@patch("nugget_rag.embedder.time.sleep")
@patch("nugget_rag.embedder.urlopen")
def test_retry_4xx_does_not_retry(mock_open, mock_sleep):
    """4xx HTTPError はリトライせず即 EmbedError。"""
    from nugget_rag.embedder import EmbedError

    exc = HTTPError(url=None, code=401, msg="Unauthorized", hdrs=None, fp=None)  # type: ignore[arg-type]
    mock_open.side_effect = exc
    with pytest.raises(EmbedError, match="401"):
        _retry_client(max_retries=3).embed(["t"])
    assert mock_open.call_count == 1


@patch("nugget_rag.embedder.time.sleep")
@patch("nugget_rag.embedder.urlopen")
def test_retry_5xx_retries(mock_open, mock_sleep):
    """5xx HTTPError はリトライする。"""
    from nugget_rag.embedder import EmbedError

    exc = HTTPError(url=None, code=503, msg="Service Unavailable", hdrs=None, fp=None)  # type: ignore[arg-type]
    mock_open.side_effect = [exc, exc, exc]
    with pytest.raises(EmbedError):
        _retry_client(max_retries=2, backoff=0.0).embed(["t"])
    assert mock_open.call_count == 3


@patch("nugget_rag.embedder.time.sleep")
@patch("nugget_rag.embedder.urlopen")
def test_retry_max_retries_zero_no_retry(mock_open, mock_sleep):
    """max_retries=0 → リトライなし、1 回のみ試行。"""
    from nugget_rag.embedder import EmbedError

    mock_open.side_effect = URLError("down")
    with pytest.raises(EmbedError):
        _retry_client(max_retries=0).embed(["t"])
    assert mock_open.call_count == 1


@patch("nugget_rag.embedder.time.sleep")
@patch("nugget_rag.embedder.urlopen")
def test_retry_backoff_called(mock_open, mock_sleep):
    """リトライ間に time.sleep が backoff * 2^attempt で呼ばれる。"""
    mock_open.side_effect = [URLError("err"), URLError("err"), _ok_response()]
    _retry_client(max_retries=3, backoff=1.0).embed(["t"])
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)  # attempt 0: 1.0 * 2^0
    mock_sleep.assert_any_call(2.0)  # attempt 1: 1.0 * 2^1
