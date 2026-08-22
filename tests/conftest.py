"""Shared test helpers."""
from __future__ import annotations


def make_results(texts: list[str], field: str = "text") -> list[dict]:
    """Build a list of result dicts with the given field set to each text."""
    return [{field: t} for t in texts]
