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


# ── Japanese / CJK tokenizer tests ──────────────────────────────────────────

def test_char_ngrams_basic():
    assert _char_ngrams("東京都") == ["東京", "京都"]


def test_char_ngrams_short():
    assert _char_ngrams("A") == ["a"]


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
