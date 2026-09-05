"""Shared test helpers."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_results(texts: list[str], field: str = "text") -> list[dict]:
    """Build a list of result dicts with the given field set to each text."""
    return [{field: t} for t in texts]


@contextmanager
def redirect_server() -> Iterator[tuple[str, list[str]]]:
    """Local HTTP server answering every request with a 302 redirect.

    Yields (base_url, hits) where hits records each request path received.
    The Location target is a closed port, so any client that follows the
    redirect would fail loudly instead of silently succeeding.
    """
    hits: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def _redirect(self) -> None:
            hits.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/steal-key")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = _redirect
        do_POST = _redirect

        def log_message(self, *args: object) -> None:  # silence test output
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
