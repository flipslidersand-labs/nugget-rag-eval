"""Thin HTTP client for the MINIPC embedding service (multilingual-e5-base).

The service accepts POST /embed/batch with {"texts": [...]} and returns
{"vectors": [[float, ...], ...]} (preferred) or {"embeddings": [[float, ...], ...]}.

Usage:
    # Set EMBEDDING_API_KEY env var before calling
    client = EmbedClient("http://<internal-host>:9092")
    vecs = client.embed(["query text", "sentence one", "sentence two"])
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import sys
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request

from nugget_rag.http_util import (
    RETRYABLE_EXCEPTIONS,
    is_retryable_http,
    read_capped,
    request_with_retry,
    urlopen,
    validate_url,
)
from nugget_rag.vectors import cosine_similarity, embed_scores  # noqa: F401 (re-exported)

MAX_BATCH_SIZE = 256
# Response body cap. A full batch is ~256 vectors x 768 dims x ~25 B of JSON text
# (about 5 MB), so 32 MiB leaves wide headroom while still bounding memory use.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def _is_plain_http_remote(url: str) -> bool:
    """True if url is http:// to a non-loopback host (X-API-Key would travel in cleartext)."""
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):  # RFC 6761
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    return not (ip.is_loopback or (mapped is not None and mapped.is_loopback))


class EmbedError(RuntimeError):
    """Raised when the embedding service call fails."""


def validate_vectors(vecs: list) -> None:
    """Raise EmbedError unless every vector is a non-empty list of finite int/float (#326).

    Messages carry only positions and type names, never the offending values.
    Dimension agreement across vectors is checked downstream (#251).
    """
    for i, vec in enumerate(vecs):
        if not isinstance(vec, list):
            raise EmbedError(
                f"Unexpected embedding response format: vector {i} must be a list, "
                f"got {type(vec).__name__}"
            )
        if not vec:
            raise EmbedError(f"Unexpected embedding response format: vector {i} is empty")
        for j, x in enumerate(vec):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise EmbedError(
                    f"Unexpected embedding response format: vector {i} element {j} "
                    f"must be a number, got {type(x).__name__}"
                )
            # float(x) overflows for huge ints, and cosine_similarity (math.sqrt) would too.
            try:
                finite = math.isfinite(float(x))
            except OverflowError:
                finite = False
            if not finite:
                raise EmbedError(
                    f"Unexpected embedding response format: vector {i} element {j} "
                    f"is not a finite float-representable number"
                )


def _check_api_key(api_key: str) -> None:
    """Reject keys with control or non-ASCII chars, without echoing the key (#303).

    http.client's own errors embed the whole header value, so a key read from a
    CRLF file would otherwise be printed to stderr / CI logs.
    """
    if any(not (0x20 <= ord(c) < 0x7F) for c in api_key):
        raise EmbedError(
            "API key contains control or non-ASCII characters (check line endings of the key file)"
        )


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        collection: str = "search-engine",
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        validate_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._warned_plain_http = False

    def _call(self, texts: list[str]) -> list[list[float]]:
        """Send a single batch request with exponential-backoff retry.

        Accepts responses with either a ``"vectors"`` or ``"embeddings"`` key so
        that the client tolerates both the current and legacy service schema.
        Retries up to max_retries times on URLError, 5xx HTTPError, or OSError/HTTPException raised while reading.
        4xx errors are raised immediately without retrying.

        Raises:
            EmbedError: after all retries exhausted, or on non-retryable errors.
        """
        api_key = os.environ.get("EMBEDDING_API_KEY", "")
        _check_api_key(api_key)
        if api_key and not self._warned_plain_http and _is_plain_http_remote(self.base_url):
            self._warned_plain_http = True
            print(
                "[WARN] X-API-Key is sent over plain http:// to a non-loopback host; "
                "use https:// to avoid exposing the key",
                file=sys.stderr,
            )
        body = json.dumps({"texts": texts, "collection": self.collection}).encode()
        req = Request(
            urljoin(self.base_url + "/", "embed/batch"),
            data=body,
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
        )

        def open_and_read() -> bytes:
            with urlopen(req, timeout=self.timeout) as http_resp:
                raw = read_capped(http_resp, MAX_RESPONSE_BYTES)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise EmbedError(
                    f"Embedding response exceeds {MAX_RESPONSE_BYTES} bytes; refusing to parse"
                )
            return raw

        try:
            raw = request_with_retry(open_and_read, self.max_retries, self.retry_backoff)
        except HTTPError as exc:
            if not is_retryable_http(exc.code):
                raise EmbedError(f"Embedding service error {exc.code}: {exc}") from exc
            raise EmbedError(
                f"Embedding service failed after {self.max_retries} retries: {exc}"
            ) from exc
        except RETRYABLE_EXCEPTIONS as exc:
            raise EmbedError(
                f"Embedding service failed after {self.max_retries} retries: {exc}"
            ) from exc
        try:
            resp = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # 応答側の不正のみ (リクエスト側の ValueError は変換しない)
            raise EmbedError(f"Unexpected embedding response format: {exc}") from exc
        if not isinstance(resp, dict):
            raise EmbedError(
                f"Unexpected embedding response format: expected object, got {type(resp).__name__}"
            )
        # Select by key presence, not truthiness: an empty "vectors" list is a
        # count mismatch, not a missing key (#329).
        vecs = resp["vectors"] if "vectors" in resp else resp.get("embeddings")
        if vecs is None:
            raise EmbedError(
                f"Expected 'vectors' or 'embeddings' key in response, got: {list(resp)}"
            )
        if not isinstance(vecs, list):
            raise EmbedError(
                f"Unexpected embedding response format: vectors must be a list, got {type(vecs).__name__}"
            )
        if len(vecs) != len(texts):
            raise EmbedError(
                f"Embedding count mismatch: sent {len(texts)} texts, got {len(vecs)} vectors"
            )
        validate_vectors(vecs)
        return vecs

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
