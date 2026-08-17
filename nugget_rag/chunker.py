"""Sentence-level splitter for nugget extraction."""

import re

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple regex-based, no external deps."""
    sentences = _SENT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]
