from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from nugget_rag.embedder import EmbedClient, cosine_similarity, embed_scores
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
