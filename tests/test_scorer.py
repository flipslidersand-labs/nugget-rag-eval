from nugget_rag.scorer import _char_ngrams, _normalize, _tokenize, bm25_scores, top_nuggets


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


def test_top_nuggets_preserves_original_order():
    sentences = [
        "First sentence about KV cache.",
        "Second sentence with unrelated content.",
        "Third sentence also about KV cache.",
        "Fourth sentence unrelated.",
        "Fifth sentence mentioning KV cache again.",
    ]
    result = top_nuggets("KV cache", sentences, top_k=3)
    assert all("KV cache" in s for s in result)
    assert result == [
        "First sentence about KV cache.",
        "Third sentence also about KV cache.",
        "Fifth sentence mentioning KV cache again.",
    ]


# ── Japanese / CJK tokenizer tests ──────────────────────────────────────────


def test_char_ngrams_basic():
    tokens = _char_ngrams("東京都")
    assert tokens == ["東京", "京都"]


def test_char_ngrams_short():
    tokens = _char_ngrams("A")
    assert tokens == ["a"]


def test_tokenize_japanese_uses_ngrams():
    tokens = _tokenize("機械学習の研究")
    assert all(len(t) == 2 for t in tokens)
    assert len(tokens) > 0


def test_tokenize_english_uses_split():
    assert _tokenize("KV cache reuse") == ["kv", "cache", "reuse"]


def test_bm25_japanese_relevant_higher():
    query = "機械学習"
    relevant = "機械学習モデルの評価手法について述べる。"
    unrelated = "料理のレシピと食材の組み合わせを紹介する。"
    scores = bm25_scores(query, [relevant, unrelated])
    assert scores[0] > scores[1]


def test_bm25_english_unchanged():
    sentences = ["KV cache reuse is the core method", "This paper uses diffusion models"]
    scores = bm25_scores("KV cache method", sentences)
    assert scores[0] > scores[1]


# ── _normalize 均一スコア (#48) ────────────────────────────────────────────


def test_normalize_uniform_returns_all_zeros():
    assert _normalize([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_normalize_single_value_returns_zero():
    assert _normalize([1.0]) == [0.0]


def test_top_nuggets_uniform_bm25_returns_k_items():
    """全スコア均一時は k 件返る（先頭 k 件）。"""
    sentences = ["A", "B", "C", "D", "E"]
    result = top_nuggets("xyznotinany", sentences, top_k=2)
    assert len(result) == 2


# ── bm25_scores 空クエリ・空センテンス (#49) ──────────────────────────────


def test_bm25_empty_query_returns_zero_scores():
    scores = bm25_scores("", ["hello world", "foo bar"])
    assert scores == [0.0, 0.0]


def test_bm25_all_empty_sentences_returns_zeros():
    scores = bm25_scores("query", ["", ""])
    assert all(s == 0.0 for s in scores)


def test_bm25_single_empty_sentence():
    scores = bm25_scores("query", [""])
    assert scores == [0.0]


# ── top_k クリップ (#51) ──────────────────────────────────────────────────


def test_top_nuggets_top_k_larger_than_sentences_returns_all():
    result = top_nuggets("query", ["A", "B"], top_k=10)
    assert sorted(result) == ["A", "B"]


def test_top_nuggets_top_k_zero_returns_empty():
    assert top_nuggets("query", ["A", "B"], top_k=0) == []


def test_top_nuggets_top_k_equals_len_returns_all():
    sentences = ["X", "Y", "Z"]
    result = top_nuggets("query", sentences, top_k=3)
    assert len(result) == 3


# ── ASCII ratio 境界 (#52) ────────────────────────────────────────────────


def test_tokenize_exactly_half_ascii_uses_bigrams():
    """ASCII 2文字 + 非ASCII 2文字 → ratio=0.5 → NOT > 0.5 → bigrams。"""
    tokens = _tokenize("AB東京")
    # ratio = 2/4 = 0.5 → bigram branch
    assert all(len(t) <= 2 for t in tokens)


def test_tokenize_above_half_ascii_uses_split():
    """ASCII 3文字 + 非ASCII 1文字 → ratio=0.75 > 0.5 → split。"""
    tokens = _tokenize("ABC東")
    assert tokens == ["abc東"]


def test_tokenize_empty_string_returns_empty():
    assert _tokenize("") == []


# ── embed_weight 極端値 (#53) ────────────────────────────────────────────


def test_top_nuggets_embed_weight_zero_equals_bm25_only():
    """embed_weight=0.0 では embed_fn を渡しても BM25-only と同じ結果。"""
    sentences = ["KV cache reuse", "diffusion model training"]

    def mock_embed(texts):
        # embedding で diffusion を高スコアに設定（BM25 と逆順）
        return [[0.0, 1.0]] + [[0.0, 1.0], [1.0, 0.0]]

    result_w0 = top_nuggets("KV", sentences, top_k=1, embed_fn=mock_embed, embed_weight=0.0)
    result_bm25 = top_nuggets("KV", sentences, top_k=1)
    assert result_w0 == result_bm25


def test_top_nuggets_embed_weight_one_uses_only_embedding():
    """embed_weight=1.0 では BM25 を無視し embedding 順になる。"""
    sentences = ["diffusion model", "KV cache reuse"]

    def mock_embed(texts):
        # query と sentences[0]（diffusion）が近い
        return [[1.0, 0.0]] + [[0.9, 0.1], [0.1, 0.9]]

    result = top_nuggets("query", sentences, top_k=1, embed_fn=mock_embed, embed_weight=1.0)
    assert result == ["diffusion model"]
