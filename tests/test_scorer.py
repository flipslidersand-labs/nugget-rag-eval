from nugget_rag.scorer import _char_ngrams, _tokenize, bm25_scores, top_nuggets


def test_bm25_returns_correct_length():
    sentences = ["KV cache reuse in RAG", "Diffusion model training", "KV cache optimization"]
    assert len(bm25_scores("KV cache", sentences)) == 3


def test_bm25_relevant_scores_higher():
    sentences = ["KV cache reuse is the core method", "This paper uses diffusion models"]
    scores = bm25_scores("KV cache method", sentences)
    assert scores[0] > scores[1]


def test_top_nuggets_returns_top_k():
    sentences = ["A", "B", "KV cache method C", "D", "KV E"]
    result = top_nuggets("KV cache", sentences, top_k=2)
    assert len(result) == 2
    assert any("KV" in s for s in result)


def test_top_nuggets_empty():
    assert top_nuggets("query", []) == []


<<<<<<< HEAD
def test_top_nuggets_preserves_original_order():
    """Test that top_nuggets returns results in original document order.

    This ensures that when multiple sentences are selected, they appear
    in the same order as they appear in the original text, preserving
    context flow and causal relationships.
    """
    sentences = [
        "First sentence about KV cache.",
        "Second sentence with unrelated content.",
        "Third sentence also about KV cache.",
        "Fourth sentence unrelated.",
        "Fifth sentence mentioning KV cache again.",
    ]
    result = top_nuggets("KV cache", sentences, top_k=3)

    # All results should mention KV cache
    assert all("KV cache" in s for s in result)

    # Results should be in original order: 0, 2, 4
    # We select top-3, which should be sentences at indices 0, 2, 4
    # After sorting by original index, they should appear as: [0, 2, 4]
    assert result == [
        "First sentence about KV cache.",
        "Third sentence also about KV cache.",
        "Fifth sentence mentioning KV cache again.",
    ]
=======
# ── Japanese / CJK tokenizer tests ──────────────────────────────────────────


def test_char_ngrams_basic():
    """_char_ngrams produces bigrams for simple Japanese text."""
    tokens = _char_ngrams("東京都")
    assert tokens == ["東京", "京都"]


def test_char_ngrams_short():
    """Text shorter than n returns individual characters."""
    tokens = _char_ngrams("A")
    assert tokens == ["a"]


def test_tokenize_japanese_uses_ngrams():
    """Japanese text (low ASCII ratio) is tokenized with character bigrams."""
    tokens = _tokenize("機械学習の研究")
    # All tokens should be 2-char strings
    assert all(len(t) == 2 for t in tokens)
    assert len(tokens) > 0


def test_tokenize_english_uses_split():
    """English text (high ASCII ratio) still uses whitespace splitting."""
    tokens = _tokenize("KV cache reuse")
    assert tokens == ["kv", "cache", "reuse"]


def test_bm25_japanese_returns_scores():
    """BM25 score computation works for Japanese query and sentences."""
    query = "機械学習"
    sentences = ["機械学習は重要な技術です。", "自然言語処理の応用例を示します。"]
    scores = bm25_scores(query, sentences)
    assert len(scores) == 2
    # Both scores should be finite numbers
    assert all(isinstance(s, float) for s in scores)


def test_bm25_japanese_relevant_higher():
    """Japanese relevant sentence scores higher than unrelated sentence."""
    query = "機械学習"
    relevant = "機械学習モデルの評価手法について述べる。"
    unrelated = "料理のレシピと食材の組み合わせを紹介する。"
    scores = bm25_scores(query, [relevant, unrelated])
    assert scores[0] > scores[1]


def test_bm25_english_unchanged():
    """English BM25 behavior is unaffected by the language detection change."""
    sentences = ["KV cache reuse is the core method", "This paper uses diffusion models"]
    scores = bm25_scores("KV cache method", sentences)
    assert scores[0] > scores[1]
>>>>>>> 9104849 (fix(scorer): 日本語向け文字 n-gram tokenizer を追加 (Closes #14))
