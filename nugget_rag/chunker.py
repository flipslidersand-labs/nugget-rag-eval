"""Sentence-level splitter for nugget extraction."""

import re

# CJK terminators split even without trailing whitespace; ASCII terminators need it,
# so decimals ("0.5", "88.3%") and "Fig.2" are not cut.
_SENT_RE = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")

# A segment ending in one of these (case-insensitive) + "." is not a sentence end (#283).
_ABBREVIATIONS = frozenset({"al", "e.g", "i.e", "fig", "eq", "sec", "vs", "cf", "etc"})
# "No." is an abbreviation only before a number ("No. 5"); "answer is no." ends a sentence.
_NUMBER_ABBREVIATIONS = frozenset({"no"})


def _ends_with_abbreviation(segment: str, following: str) -> bool:
    if not segment.endswith("."):
        return False
    words = segment[:-1].split()
    last = words[-1].lower() if words else ""
    if last in _ABBREVIATIONS:
        return True
    return last in _NUMBER_ABBREVIATIONS and following[:1].isdigit()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple regex-based, no external deps."""
    pieces = [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ends_with_abbreviation(sentences[-1], piece):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences
