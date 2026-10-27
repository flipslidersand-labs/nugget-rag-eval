"""Sentence-level splitter for nugget extraction."""

import re

# CJK terminators split even without trailing whitespace; ASCII terminators need it,
# so decimals ("0.5", "88.3%") and "Fig.2" are not cut.
_SENT_RE = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple regex-based, no external deps."""
    sentences = _SENT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]
