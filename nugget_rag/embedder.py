"""Thin HTTP client for the MINIPC embedding service (multilingual-e5-base).

The service accepts POST /embed/batch with {"texts": [...]} and returns
{"embeddings": [[float, ...], ...]}.

Usage:
    # Set EMBEDDING_API_KEY env var before calling
    client = EmbedClient("http://<internal-host>:9092")
    vecs = client.embed(["query text", "sentence one", "sentence two"])
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
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
        api_key: str | None = None,
        collection: str = "search-engine",
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        _validate_url(base_url)
        self.base_url = base_url.rstrip("/")
        if api_key is not None:
            warnings.warn(
                "Passing api_key to EmbedClient is deprecated and has no effect. "
                "Set the EMBEDDING_API_KEY environment variable instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.collection = collection
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _call(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch request with exponential-backoff retry.

        Retries up to max_retries times on URLError or 5xx HTTPError.
        4xx errors are raised immediately without retrying.

        Raises:
            EmbedError: after all retries exhausted, or on non-retryable errors.
        """
        api_key = os.environ.get("EMBEDDING_API_KEY", "")
        body = json.dumps({"texts": texts, "collection": self.collection}).encode()
        req = Request(
            urljoin(self.base_url + "/", "embed/batch"),
            data=body,
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = json.loads(urlopen(req, timeout=self.timeout).read())
                return resp["vectors"]
            except HTTPError as exc:
                if exc.code is not None and exc.code < 500:
                    raise EmbedError(f"Embedding service error {exc.code}: {exc}") from exc
                last_exc = exc
            except URLError as exc:
                last_exc = exc
            except (json.JSONDecodeError, KeyError) as exc:
                raise EmbedError(f"Unexpected embedding response format: {exc}") from exc
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * (2**attempt))
        raise EmbedError(
            f"Embedding service unreachable after {self.max_retries} retries: {last_exc}"
        ) from last_exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each text.

        Automatically splits inputs into batches of MAX_BATCH_SIZE to respect
        the MINIPC embedding-svc per-request limit. Each batch is retried
        independently via _call().

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
