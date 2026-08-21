"""Tests for --token-estimator option (words/chars modes) in avg_tokens."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure eval/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from evaluate import avg_tokens


def _make_results(texts: list[str], field: str = "text") -> list[dict]:
    return [{field: t} for t in texts]


def test_words_mode_english():
    results = _make_results(["hello world foo", "bar baz"])
    # item 0: 3 words, item 1: 2 words → avg = (3+2)/2 = 2.5
    assert avg_tokens(results, field="text", estimator="words") == 2.5


def test_chars_mode_basic():
    results = _make_results(["abcdefgh"])  # 8 chars → 8/4 = 2.0
    assert avg_tokens(results, field="text", estimator="chars") == 2.0


def test_chars_mode_formula():
    texts = ["aaaa", "aaaaaaaa"]  # 4 chars → 1.0, 8 chars → 2.0 → avg = 1.5
    results = _make_results(texts)
    assert avg_tokens(results, field="text", estimator="chars") == 1.5


def test_japanese_chars_greater_than_words():
    """日本語テキストでは chars モードが words モードより高い値を返す。"""
    japanese = "日本語のテキストはスペースで区切られないため split では1単語になる"
    results = _make_results([japanese])
    words_val = avg_tokens(results, field="text", estimator="words")
    chars_val = avg_tokens(results, field="text", estimator="chars")
    assert chars_val > words_val, (
        f"Expected chars ({chars_val}) > words ({words_val}) for Japanese text"
    )


def test_empty_results():
    assert avg_tokens([], field="text", estimator="words") == 0.0
    assert avg_tokens([], field="text", estimator="chars") == 0.0


def test_default_estimator_is_words():
    results = _make_results(["one two three"])
    assert avg_tokens(results) == 3.0
