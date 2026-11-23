"""Unit tests for nugget_rag.http_util (shared retry loop). No network; sleep is patched."""

import http.client
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from nugget_rag.http_util import (
    is_retryable_http,
    read_capped,
    request_with_retry,
    retry_delay,
    urlopen,
)


def _http_error(code, retry_after=None):
    hdrs = {"Retry-After": retry_after} if retry_after is not None else {}
    return HTTPError("http://x/", code, "err", hdrs, None)


def _flaky(*outcomes):
    """open_fn that returns/raises each outcome in turn."""
    it = iter(outcomes)

    def fn():
        o = next(it)
        if isinstance(o, BaseException):
            raise o
        return o

    return fn


@pytest.mark.parametrize("code", [500, 503, 408, 429, None])
def test_is_retryable_true(code):
    assert is_retryable_http(code) is True


@pytest.mark.parametrize("code", [301, 400, 401, 404])
def test_is_retryable_false(code):
    assert is_retryable_http(code) is False


@pytest.mark.parametrize(
    ("header", "expected"),
    [("7", 7.0), ("999", 30), ("0", 1.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 1.0), ("-5", 1.0)],
)
def test_retry_delay_honours_capped_integer_retry_after(header, expected):
    assert retry_delay(_http_error(503, header), 1.0) == expected


def test_retry_delay_without_header_is_backoff():
    assert retry_delay(_http_error(503), 2.0) == 2.0


@patch("nugget_rag.http_util.time.sleep")
def test_returns_first_success_without_sleeping(sleep):
    assert request_with_retry(_flaky(b"ok"), 3, 1.0) == b"ok"
    sleep.assert_not_called()


@patch("nugget_rag.http_util.time.sleep")
def test_retries_with_exponential_delays_then_succeeds(sleep):
    fn = _flaky(URLError("down"), _http_error(500), TimeoutError("t"), b"ok")
    assert request_with_retry(fn, 3, 1.0) == b"ok"
    assert [c.args[0] for c in sleep.call_args_list] == [1.0, 2.0, 4.0]


@patch("nugget_rag.http_util.time.sleep")
def test_exhaustion_reraises_last_exception_after_max_retries_plus_one_calls(sleep):
    calls = []
    last = URLError("last")

    def fn():
        calls.append(1)
        raise URLError("first") if len(calls) == 1 else last

    with pytest.raises(URLError) as ei:
        request_with_retry(fn, 2, 0.5)
    assert ei.value is last
    assert len(calls) == 3
    assert [c.args[0] for c in sleep.call_args_list] == [0.5, 1.0]  # no sleep after last


@patch("nugget_rag.http_util.time.sleep")
def test_zero_retries_calls_once_and_does_not_sleep(sleep):
    with pytest.raises(URLError):
        request_with_retry(_flaky(URLError("x")), 0, 1.0)
    sleep.assert_not_called()


@patch("nugget_rag.http_util.time.sleep")
def test_retry_after_raises_delay_but_is_capped(sleep):
    fn = _flaky(_http_error(429, "5"), _http_error(503, "9999"), b"ok")
    assert request_with_retry(fn, 3, 1.0) == b"ok"
    assert [c.args[0] for c in sleep.call_args_list] == [5, 30]


@pytest.mark.parametrize("code", [400, 404])
@patch("nugget_rag.http_util.time.sleep")
def test_non_retryable_http_error_is_raised_closed_without_retry(sleep, code):
    exc = _http_error(code)
    exc.close = MagicMock()
    with pytest.raises(HTTPError) as ei:
        request_with_retry(_flaky(exc, b"never"), 3, 1.0)
    assert ei.value is exc
    exc.close.assert_called_once()
    sleep.assert_not_called()


@patch("nugget_rag.http_util.time.sleep")
def test_retryable_http_error_is_closed_before_each_retry(sleep):
    first, last = _http_error(503), _http_error(503)
    first.close, last.close = MagicMock(), MagicMock()
    with pytest.raises(HTTPError) as ei:
        request_with_retry(_flaky(first, last), 1, 1.0)
    assert ei.value is last
    first.close.assert_called_once()
    last.close.assert_called_once()


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("t"),
        ConnectionResetError("r"),
        http.client.IncompleteRead(b"x", 5),
        http.client.RemoteDisconnected("d"),
    ],
)
@patch("nugget_rag.http_util.time.sleep")
def test_read_time_errors_are_retried(sleep, exc):
    assert request_with_retry(_flaky(exc, b"ok"), 1, 1.0) == b"ok"
    sleep.assert_called_once_with(1.0)


@patch("nugget_rag.http_util.time.sleep")
def test_unlisted_exception_propagates_without_retry(sleep):
    with pytest.raises(KeyError):
        request_with_retry(_flaky(KeyError("k"), b"never"), 3, 1.0)
    sleep.assert_not_called()


@patch("nugget_rag.http_util.time.sleep")
def test_retryable_exceptions_are_configurable(sleep):
    assert request_with_retry(_flaky(ValueError("v"), b"ok"), 1, 1.0, (ValueError,)) == b"ok"
    with pytest.raises(TimeoutError):  # not in the given set: propagates at once
        request_with_retry(_flaky(TimeoutError("t"), b"never"), 3, 1.0, (ValueError,))


def _resp(body, content_length=None):
    resp = MagicMock()
    resp.read.return_value = body
    resp.headers = {} if content_length is None else {"Content-Length": content_length}
    return resp


def test_read_capped_reads_one_past_the_cap():
    resp = _resp(b"abc")
    assert read_capped(resp, 10) == b"abc"
    resp.read.assert_called_once_with(11)


def test_read_capped_raises_incomplete_read_on_short_body():
    with pytest.raises(http.client.IncompleteRead):
        read_capped(_resp(b"abc", "10"), 100)


def test_read_capped_ignores_bad_or_oversized_content_length():
    assert read_capped(_resp(b"abc", "x"), 100) == b"abc"
    assert read_capped(_resp(b"abc", "1000"), 100) == b"abc"  # declared > cap: not a truncation


def test_urlopen_refuses_redirects():
    from nugget_rag.http_util import _NoRedirectHandler

    with pytest.raises(HTTPError, match="Redirect to .* refused"):
        _NoRedirectHandler().redirect_request(
            MagicMock(full_url="http://a/"), None, 302, "Found", {}, "http://evil/"
        )
    assert callable(urlopen)
