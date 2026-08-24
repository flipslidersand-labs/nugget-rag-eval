"""Thin HTTP client for the MINIPC embedding service (multilingual-e5-base).

The service accepts POST /embed/batch with {"texts": [...]} and returns
{"embeddings": [[float, ...], ...]}.

Usage:
    client = EmbedClient("http://<internal-host>:9092", api_key="...")
    vecs = client.embed(["query text", "sentence one", "sentence two"])
"""

from __future__ import annotations

import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

MAX_BATCH_SIZE = 256

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _validate_url(url: str) -> None:
    """Reject non-HTTP(S) schemes to prevent SSRF via file://, ftp://, etc."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme!r}. Only 'http' and 'https' are allowed."
        )


class EmbedError(RuntimeError):
    """Raised when the embedding service call fails."""


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        collection: str = "search-engine",
        timeout: int = 60,
    ) -> None:
        _validate_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.collection = collection
        self.timeout = timeout

    def _call(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch request (must be <= MAX_BATCH_SIZE)."""
        body = json.dumps({"texts": texts, "collection": self.collection}).encode()
        req = Request(
            urljoin(self.base_url + "/", "embed/batch"),
            data=body,
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
        )
        try:
            resp = json.loads(urlopen(req, timeout=self.timeout).read())
            return resp["vectors"]
        except (URLError, HTTPError) as exc:
            raise EmbedError(f"Embedding service unreachable: {exc}") from exc
        except (json.JSONDecodeError, KeyError) as exc:
            raise EmbedError(f"Unexpected embedding response format: {exc}") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each text.

        Automatically splits inputs into batches of MAX_BATCH_SIZE to respect
        the MINIPC embedding-svc per-request limit.

        Raises:
            EmbedError: on network failure, HTTP error, or unexpected response format.
        """
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            results.extend(self._call(texts[i : i + MAX_BATCH_SIZE]))
        return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_scores(query_vec: list[float], sentence_vecs: list[list[float]]) -> list[float]:
    """Return cosine similarity of each sentence vector against the query vector."""
    return [cosine_similarity(query_vec, sv) for sv in sentence_vecs]
