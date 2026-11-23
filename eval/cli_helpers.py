"""Shared argparse ``type=`` helpers and JSON loader for the CLI entry points (#175).

Public names on purpose: ``evaluate.py``, ``check_regression.py``,
``build_pool_from_pdfs.py`` and ``fetch_papers.py`` all import from here, so no
module reaches into another module's underscore-prefixed functions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nugget_rag.embedder import EmbedClient


def float_between_0_1(v: str) -> float:
    """Argparse type: a float in [0.0, 1.0]."""
    f = float(v)
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {f}")
    return f


def positive_int(v: str) -> int:
    """Argparse type: an integer >= 1."""
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def non_negative_int(v: str) -> int:
    """Argparse type: an integer >= 0."""
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def existing_file(value: str) -> str:
    """Argparse type: abort early if the path does not point to a file."""
    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"file not found: {value}")
    return value


def add_embedding_args(
    parser: argparse.ArgumentParser, *, weight: bool = True, url_required: bool = False
) -> None:
    """Add the embedding-service options shared by ``evaluate.py`` and the sweep (#334).

    ``weight=False`` omits ``--embed-weight`` (the sweep has its own ``--weights``).
    ``url_required`` makes ``--embedding-url`` mandatory (the sweep cannot run without it).
    """
    parser.add_argument(
        "--embedding-url",
        default=None,
        required=url_required,
        help="Embedding service URL (e.g. http://<internal-host>:9092). "
        "When set, nugget scoring uses BM25 + embedding hybrid.",
    )
    if weight:
        parser.add_argument(
            "--embed-weight",
            type=float_between_0_1,
            default=0.5,
            help="Embedding weight in hybrid score: 0=BM25-only, 1=embed-only (default: 0.5)",
        )
    parser.add_argument(
        "--embedding-collection",
        default="search-engine",
        help="Embedding collection name (default: search-engine)",
    )


def build_embed_client(args: argparse.Namespace) -> EmbedClient:
    """Build the EmbedClient from ``add_embedding_args`` options.

    Warns on stderr when ``EMBEDDING_API_KEY`` is unset; an invalid URL becomes
    ``sys.exit("[ERROR] --embedding-url: ...")`` instead of a ValueError traceback.
    """
    from nugget_rag.embedder import EmbedClient  # lazy: keeps plain BM25 runs import-light

    if not os.environ.get("EMBEDDING_API_KEY"):
        print("[WARN] EMBEDDING_API_KEY not set", file=sys.stderr)
    try:
        return EmbedClient(args.embedding_url, collection=args.embedding_collection)
    except ValueError as exc:
        sys.exit(f"[ERROR] --embedding-url: {exc}")


def load_json(path: str, label: str) -> list[dict]:
    """Read *path* and parse JSON, exiting with a friendly [ERROR] message on failure."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[ERROR] {label} file not found: {path}")
    except PermissionError:
        sys.exit(f"[ERROR] permission denied reading {label} file: {path}")
    except UnicodeDecodeError:
        sys.exit(f"[ERROR] {label} file is not valid UTF-8: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"[ERROR] {label} file is not valid JSON ({path}): {exc}")


def write_json_atomic(
    path: Path, data: object, *, indent: int | None = 2, trailing_newline: bool = False
) -> None:
    """Write *data* as UTF-8 JSON (``ensure_ascii=False``) to *path* atomically.

    Serialises first, writes a sibling tmp file (``<name>.<pid>.tmp``, so a *path* that already has
    an extension or a same-stem sibling is never touched), then ``os.replace``s it. On any failure
    the tmp file is removed and an existing *path* is left as it was.
    """
    text = json.dumps(data, ensure_ascii=False, indent=indent)
    if trailing_newline:
        text += "\n"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
