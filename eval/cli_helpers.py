"""Shared argparse ``type=`` helpers and JSON loader for the CLI entry points (#175).

Public names on purpose: ``evaluate.py``, ``check_regression.py``,
``build_pool_from_pdfs.py`` and ``fetch_papers.py`` all import from here, so no
module reaches into another module's underscore-prefixed functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
