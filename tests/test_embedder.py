import json as _json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from nugget_rag.embedder import (
    MAX_BATCH_SIZE,
    EmbedClient,
    _validate_url,
    cosine_similarity,
    embed_scores,
)
from nugget_rag.scorer import top_nuggets


def _make_cm_mock(data: bytes) -> MagicMock:
    """Return a urlopen callable mock that acts as a context manager yielding a resp with .read()."""
    resp = MagicMock()
    resp.read.return_value = data
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=cm)


def _cm_side_effect(fn):
    """Wrap a side_effect fn(req, timeout)->bytes so urlopen acts as context manager."""

    def wrapper(req, timeout):
        data = fn(req, timeout)
        resp = MagicMock()
        resp.read.return_value = data
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=resp)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    return wrapper


# --- URL スキーム検証 (SSRF 対策) ---


def test_validate_url_allows_http():
    _validate_url("http://localhost:9092")  # should not raise


def test_validate_url_allows_https():
    _validate_url("https://example.com/embed")  # should not raise


def test_validate_url_rejects_file_scheme():
    with pytest.raises(ValueError, match="file"):
        _validate_url("file:///etc/passwd")


def test_validate_url_rejects_ftp_scheme():
    with pytest.raises(ValueError, match="ftp"):
        _validate_url("ftp://internal-host/resource")


def test_validate_url_rejects_imds_url():
    """AWS IMDSv1 経由の SSRF を防ぐ。スキームが http でも検証済みなら通過するが、
    ここではカスタムスキーム的な悪用パターンを確認する。"""
    # file:// / ftp:// は拒否される
    with pytest.raises(ValueError):
        _validate_url("file://169.254.169.254/latest/meta-data/")


def test_embed_client_rejects_file_scheme_base_url():
    with pytest.raises(ValueError, match="file"):
        EmbedClient("file:///etc/passwd", api_key="k")


def test_embed_client_rejects_ftp_scheme_base_url():
    with pytest.raises(ValueError, match="ftp"):
        EmbedClient("ftp://internal/resource", api_key="k")


def test_embed_client_accepts_http_base_url():
    client = EmbedClient("http://localhost:9092", api_key="k")
    assert client.base_url == "http://localhost:9092"


def test_embed_client_accepts_https_base_url():
    client = EmbedClient("https://embed.example.com", api_key="k")
    assert client.base_url == "https://embed.example.com"


# --- EmbedClient.embed() エラーパス ---


def _client():
    return EmbedClient("http://localhost:9092")


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


def test_embed_unknown_key_raises_embed_error_with_detail():
    """Response with neither 'vectors' nor 'embeddings' raises EmbedError with key list."""
    from nugget_rag.embedder import EmbedError

    with patch("nugget_rag.embedder.urlopen", _make_cm_mock(b'{"data": [[1.0, 2.0]]}')):
        with pytest.raises(EmbedError, match="'vectors' or 'embeddings'"):
            _client().embed(["hello"])


def test_embed_embeddings_key_fallback_succeeds():
    """Response with 'embeddings' key (legacy schema) is accepted transparently."""
    with patch("nugget_rag.embedder.urlopen", _make_cm_mock(b'{"embeddings": [[0.1, 0.2]]}')):
        result = _client().embed(["hello"])
    assert result == [[0.1, 0.2]]


def test_embed_count_mismatch_raises_embed_error():
    """Mismatched vector count raises EmbedError with a descriptive message."""
    from nugget_rag.embedder import EmbedError

    with patch(
        "nugget_rag.embedder.urlopen", _make_cm_mock(b'{"vectors": [[0.1, 0.2], [0.3, 0.4]]}')
    ):
        # Sending 1 text but receiving 2 vectors should raise
        with pytest.raises(EmbedError, match="mismatch"):
            _client().embed(["only one text"])


def test_embed_invalid_json_raises():
    with patch("nugget_rag.embedder.urlopen", _make_cm_mock(b"not json")):
        with pytest.raises(Exception):
            _client().embed(["hello"])


def test_embed_success_returns_vectors():
    with patch(
        "nugget_rag.embedder.urlopen", _make_cm_mock(b'{"vectors": [[0.1, 0.2], [0.3, 0.4]]}')
    ):
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

    def raw_side_effect(req, timeout):
        batch_size = len(_json.loads(req.data)["texts"])
        return _json.dumps({"vectors": [[0.1, 0.2]] * batch_size}).encode()

    mock_open.side_effect = _cm_side_effect(raw_side_effect)

    n = MAX_BATCH_SIZE + 10
    vecs = _client().embed(["text"] * n)
    assert len(vecs) == n
    assert mock_open.call_count == 2  # 256 + 10


@patch("nugget_rag.embedder.urlopen")
def test_embed_exact_batch_size_single_call(mock_open):
    """texts == MAX_BATCH_SIZE のとき urlopen は 1 回だけ。"""
    data = _json.dumps({"vectors": [[0.0]] * MAX_BATCH_SIZE}).encode()
    mock_open.side_effect = _cm_side_effect(lambda req, timeout: data)
    vecs = _client().embed(["t"] * MAX_BATCH_SIZE)
    assert len(vecs) == MAX_BATCH_SIZE
    assert mock_open.call_count == 1


@patch("nugget_rag.embedder.urlopen")
def test_embed_batch_preserves_order(mock_open):
    """複数バッチでもベクトルの順序が保たれる。"""
    call_idx = [0]

    def raw_side_effect(req, timeout):
        batch = _json.loads(req.data)["texts"]
        vecs = [[float(i + call_idx[0] * MAX_BATCH_SIZE)] for i in range(len(batch))]
        call_idx[0] += 1
        return _json.dumps({"vectors": vecs}).encode()

    mock_open.side_effect = _cm_side_effect(raw_side_effect)
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
    data = _json.dumps({"vectors": [[0.1]] * n}).encode()
    resp = MagicMock()
    resp.read.return_value = data
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=resp)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


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
