"""HTTP building blocks shared by the embedding client and scripts/fetch_papers.py.

URL validation, a urlopen that never follows redirects, the retry policy
(which statuses and exceptions are transient, Retry-After, exponential backoff)
and the response-body reader. Callers only translate exceptions into their own
error type (EmbedError / FetchError). Standard library only (ADR-002).
"""

from __future__ import annotations

import http.client
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener

_ALLOWED_SCHEMES = frozenset({"http", "https"})

_RETRY_AFTER_MAX = 30  # seconds; upper bound honoured from a Retry-After header
_RETRYABLE_4XX = frozenset({408, 429})

# Errors raised by read() after the headers arrived (timeout / disconnect / truncated body).
READ_ERRORS = (TimeoutError, ConnectionError, http.client.HTTPException)
# HTTPError is a URLError subclass; request_with_retry handles it first.
RETRYABLE_EXCEPTIONS = (URLError, *READ_ERRORS)


def validate_url(url: str) -> None:
    """Reject non-HTTP(S) schemes, URLs without a host, and URLs with userinfo.

    This pins the scheme (blocks file://, ftp://, etc.), requires a non-empty
    host, and refuses ``user:pass@host`` (credentials belong in EMBEDDING_API_KEY).
    Error messages never echo the URL, since it may carry secrets. It does NOT
    restrict target hosts: the client is meant
    for LAN-internal services, so internal IPs are intentionally allowed.
    Redirect following is separately disabled (see _NoRedirectHandler) so
    request headers can never be forwarded beyond the validated initial URL.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme!r}. Only 'http' and 'https' are allowed."
        )
    if "@" in parsed.netloc:
        raise ValueError("URL must not contain userinfo (user:pass@host)")
    if not parsed.netloc:
        raise ValueError("URL has no host")


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


def is_retryable_http(code: int | None) -> bool:
    """True for 5xx and the transient 4xx codes 408/429 (#284)."""
    return code is None or code >= 500 or code in _RETRYABLE_4XX


def retry_delay(exc: HTTPError, backoff_delay: float) -> float:
    """Backoff delay, raised to Retry-After (integer seconds only, capped) if present.

    Non-integer / negative values (e.g. HTTP-date) are ignored.
    """
    hdrs = getattr(exc, "headers", None)
    raw = hdrs.get("Retry-After") if hdrs is not None else None
    value = raw.strip() if isinstance(raw, str) else ""
    # len <= 4 guards int() against huge digit strings (ValueError escapes the except clause)
    if value.isascii() and value.isdigit() and len(value) <= 4:
        return max(backoff_delay, min(int(value), _RETRY_AFTER_MAX))
    return backoff_delay


def read_capped(http_resp, max_bytes: int) -> bytes:
    """Read at most ``max_bytes + 1`` bytes; the caller rejects a result longer than max_bytes.

    read(amt) does not raise IncompleteRead on a short body (read() does), so a body
    shorter than its Content-Length is raised here so that it is retried (#232).
    """
    raw = http_resp.read(max_bytes + 1)
    declared = http_resp.headers.get("Content-Length")
    if (
        isinstance(declared, str)
        and declared.isascii()
        and declared.isdigit()
        and len(declared) <= 12
        and len(raw) < int(declared) <= max_bytes
    ):
        raise http.client.IncompleteRead(raw, int(declared) - len(raw))
    return raw


def request_with_retry(
    open_fn: Callable[[], bytes],
    max_retries: int,
    backoff: float,
    retryable_exceptions: tuple[type[BaseException], ...] = RETRYABLE_EXCEPTIONS,
) -> bytes:
    """Call *open_fn* (open + read the body) with exponential-backoff retry.

    Waits ``backoff * 2**attempt`` between attempts, raised to Retry-After (capped)
    on an HTTPError. Retries up to *max_retries* times on a retryable HTTPError
    (5xx/408/429) or any of *retryable_exceptions*. Any other exception, including
    a non-retryable HTTPError (closed first), propagates unchanged. When retries
    are exhausted the last exception is re-raised.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        delay = backoff * (2**attempt)
        try:
            return open_fn()
        except HTTPError as exc:
            if not is_retryable_http(exc.code):
                exc.close()  # release the response socket before re-raising (#313)
                raise
            last_exc = exc
            delay = retry_delay(exc, delay)
            exc.close()  # release the response socket before retrying / final raise (#141, #280)
        except retryable_exceptions as exc:
            last_exc = exc
        if attempt < max_retries:
            time.sleep(delay)
    assert last_exc is not None  # loop always runs at least once
    raise last_exc
