import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from evaluate import avg_tokens


def test_avg_tokens_words_default():
    results = [{"text": "KV cache reuse"}, {"text": "hello world"}]
    assert avg_tokens(results) == 2.5


def test_avg_tokens_words_explicit():
    results = [{"text": "one two three four five"}]
    assert avg_tokens(results, estimator="words") == 5.0


def test_avg_tokens_chars():
    results = [{"text": "abcd"}]  # len=4 → 4/4=1.0
    assert avg_tokens(results, estimator="chars") == 1.0


def test_avg_tokens_chars_multiple():
    results = [{"text": "abcdefgh"}, {"text": "abcdefgh"}]  # each 8/4=2.0
    assert avg_tokens(results, estimator="chars") == 2.0


def test_avg_tokens_empty():
    assert avg_tokens([]) == 0.0
