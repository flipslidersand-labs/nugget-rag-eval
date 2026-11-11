"""Thin HTTP client for the MINIPC embedding service (multilingual-e5-base).

The service accepts POST /embed/batch with {"texts": [...]} and returns
{"vectors": [[float, ...], ...]} (preferred) or {"embeddings": [[float, ...], ...]}.

Usage:
    # Set EMBEDDING_API_KEY env var before calling
    client = EmbedClient("http://<internal-host>:9092")
    vecs = client.embed(["query text", "sentence one", "sentence two"])
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from nugget_rag.vectors import cosine_similarity, embed_scores  # noqa: F401 (re-exported)

MAX_BATCH_SIZE = 256

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _validate_url(url: str) -> None:
    """Reject non-HTTP(S) schemes and URLs without a host.

    This only pins the scheme (blocks file://, ftp://, etc.) and requires a
    non-empty netloc. It does NOT restrict target hosts: the client is meant
    for LAN-internal services, so internal IPs are intentionally allowed.
    Redirect following is separately disabled (see _NoRedirectHandler) so
    request headers can never be forwarded beyond the validated initial URL.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme!r}. Only 'http' and 'https' are allowed."
        )
    if not parsed.netloc:
        raise ValueError(f"URL has no host: {url!r}")


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


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse HTTP redirects.

    The default opener re-sends all request headers (including X-API-Key) to
    the redirect target, which may be a different host, a plaintext downgrade,
    or an attacker-controlled URL. Raising here turns any 3xx response into an
    HTTPError that callers surface as EmbedError/FetchError.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(
            req.full_url,
            code,
            f"Redirect to {newurl!r} refused: redirects are disabled to avoid "
            "forwarding credentials to unverified hosts",
            headers,
            fp,
        )


# Drop-in replacement for urllib.request.urlopen that never follows redirects.
urlopen = build_opener(_NoRedirectHandler).open


_RETRY_AFTER_MAX = 30  # seconds; upper bound honoured from a Retry-After header
_RETRYABLE_4XX = frozenset({408, 429})


def _is_retryable_http(code: int | None) -> bool:
    """True for 5xx and the transient 4xx codes 408/429 (#284). Shared with fetch_papers."""
    return code is None or code >= 500 or code in _RETRYABLE_4XX


def _retry_delay(exc: HTTPError, backoff_delay: float) -> float:
    """Backoff delay, raised to Retry-After (integer seconds only, capped) if present.

    Non-integer / negative values (e.g. HTTP-date) are ignored.
    """
    hdrs = getattr(exc, "headers", None)
    raw = hdrs.get("Retry-After") if hdrs is not None else None
    if isinstance(raw, str) and raw.strip().isascii() and raw.strip().isdigit():
        return max(backoff_delay, min(int(raw.strip()), _RETRY_AFTER_MAX))
    return backoff_delay


class EmbedError(RuntimeError):
    """Raised when the embedding service call fails."""


class EmbedClient:
    def __init__(
        self,
        base_url: str,
        collection: str = "search-engine",
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        _validate_url(base_url)
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
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            delay = self.retry_backoff * (2**attempt)
            try:
                with urlopen(req, timeout=self.timeout) as http_resp:
                    resp = json.loads(http_resp.read())
                if not isinstance(resp, dict):
                    raise EmbedError(
                        f"Unexpected embedding response format: expected object, got {type(resp).__name__}"
                    )
                vecs = resp.get("vectors") or resp.get("embeddings")
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
                return vecs
            except HTTPError as exc:
                if not _is_retryable_http(exc.code):
                    raise EmbedError(f"Embedding service error {exc.code}: {exc}") from exc
                delay = _retry_delay(exc, delay)
                exc.close()  # release the response socket before retrying (#141)
                last_exc = exc
            except URLError as exc:
                last_exc = exc
            except (TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                # read() 中のタイムアウト/切断 (HTTPError/URLError は上で捕捉済み)
                last_exc = exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # 応答側の不正のみ (リクエスト側の ValueError は変換しない)
                raise EmbedError(f"Unexpected embedding response format: {exc}") from exc
            if attempt < self.max_retries:
                time.sleep(delay)
        raise EmbedError(
            f"Embedding service failed after {self.max_retries} retries: {last_exc}"
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
