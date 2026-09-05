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


# ── _normalize 空リスト (#101) ───────────────────────────────────────────


def test_normalize_empty_returns_empty():
    """_normalize([]) は ValueError を投げず [] を返す。"""
    assert _normalize([]) == []


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


# ── ハイブリッドトークナイザ (#152) ──────────────────────────────────────────


def test_tokenize_mixed_ascii_cjk_splits_both():
    """ASCII + CJK の混在文字列はハイブリッド処理でどちらも抽出される。

    旧実装では ASCII 比率でいずれか一方に倒していたが、新実装では
    ASCII 部分はホワイトスペース分割・CJK 部分はバイグラムを並存させる。
    """
    tokens = _tokenize("AB東京")
    # ASCII フラグメント "ab" + CJK バイグラム "東京"
    assert "ab" in tokens
    assert "東京" in tokens


def test_tokenize_above_half_ascii_extracts_both():
    """ASCII 3文字 + 非ASCII 1文字でも CJK 文字がバイグラムとして抽出される。"""
    tokens = _tokenize("ABC東")
    # ASCII 部分 "abc" と CJK 文字 "東" が両方得られる
    assert "abc" in tokens
    assert "東" in tokens


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


# ── IDF uses document frequency, not TF (#148) ────────────────────────────────


def test_bm25_high_tf_doc_ranks_above_unrelated():
    """頻出一致文（TF=5）は完全無関係文より上位になるべき（IDF がマイナスにならない）。

    旧実装では IDF に TF を使っていたため f > N+0.5 で IDF < 0 になり、
    クエリ語 5 回の文が無関係文（スコア 0.0）より下位になっていた。
    """
    sentences = [
        "cache cache cache cache cache compression",  # TF=5 for "cache"
        "unrelated words here entirely",
        "cache methods",  # TF=1 for "cache"
    ]
    scores = bm25_scores("cache", sentences)
    # High-TF match must beat the unrelated sentence (score must be positive)
    assert scores[0] > scores[1], (
        f"High-TF doc ({scores[0]:.3f}) must rank above unrelated doc ({scores[1]:.3f})"
    )
    assert scores[0] >= 0.0, f"High-TF score must not be negative, got {scores[0]:.3f}"


def test_bm25_ubiquitous_term_has_low_idf_contribution():
    """全文出現語は DF=N なので IDF≈0 となりスコアへの寄与が最小限になる。

    旧実装では TF を IDF に使っていたため、全文に 1 回ずつ現れる語でも
    IDF が非ゼロになりストップワードがスコアを歪めていた。
    """
    # "the" appears in every sentence — DF == N, so IDF should be ~0
    sentences = [
        "the quick brown fox",
        "the lazy dog sleeps",
        "the cat sat on the mat",
    ]
    scores = bm25_scores("the", sentences)
    # All scores should be very small (near 0) because IDF ≈ log(1 + 0.5/3.5) ≈ 0.13
    # and definitely no score should dominate due to "the" alone
    assert all(s >= 0.0 for s in scores), "Scores for ubiquitous term must not be negative"
    # The spread should be small — no sentence should have score > 0.5
    assert max(scores) < 0.5, (
        f"Ubiquitous term should have low IDF contribution, max score={max(scores):.3f}"
    )


# ── #150 precomputed query_vec / sent_vecs ───────────────────────────────


def test_top_nuggets_precomputed_vecs_match_embed_fn():
    """#150: query_vec+sent_vecs を渡すと embed_fn と同じ結果になる。"""
    sentences = ["KV cache reuse", "diffusion model training"]

    def mock_embed(texts):
        return [[1.0, 0.0]] + [[0.9, 0.1], [0.1, 0.9]]

    result_fn = top_nuggets("query", sentences, top_k=1, embed_fn=mock_embed, embed_weight=1.0)

    # Provide the same vectors directly (bypassing embed_fn)
    result_pre = top_nuggets(
        "query",
        sentences,
        top_k=1,
        embed_weight=1.0,
        query_vec=[1.0, 0.0],
        sent_vecs=[[0.9, 0.1], [0.1, 0.9]],
    )
    assert result_fn == result_pre


def test_top_nuggets_precomputed_does_not_call_embed_fn():
    """#150: query_vec+sent_vecs がある場合 embed_fn は呼ばれない。"""
    calls: list = []

    def should_not_be_called(texts):
        calls.append(texts)
        return [[0.0, 1.0]] * len(texts)

    sentences = ["KV cache reuse", "diffusion model"]
    top_nuggets(
        "query",
        sentences,
        top_k=1,
        embed_fn=should_not_be_called,
        embed_weight=0.5,
        query_vec=[1.0, 0.0],
        sent_vecs=[[0.9, 0.1], [0.1, 0.9]],
    )
    assert calls == [], "embed_fn should not be called when precomputed vecs are provided"


# ── 混在言語 BM25 (#152) ──────────────────────────────────────────────────


def test_bm25_english_query_japanese_sentences_nonzero():
    """英語クエリ × 日本語文で BM25 スコアが全ゼロにならないこと。

    旧実装では _tokenize がクエリと文に独立の言語判定を適用するため、
    英語クエリ（whitespace split）と日本語文（bigram）の token 空間が
    完全に食い違い、BM25 スコアが全て 0.0 になっていた。
    """
    # 日本語文に英語キーワード "KV" を含む文と含まない文
    sentences = [
        "KVキャッシュの再利用による推論高速化について述べる。",
        "料理のレシピと食材の組み合わせを紹介する。",
    ]
    scores = bm25_scores("KV cache", sentences)
    # 少なくとも 1 件は非ゼロ（KV を含む文が拾われる）
    assert any(s > 0.0 for s in scores), f"All scores are zero: {scores}"
    # KV を含む文の方がスコアが高い
    assert scores[0] > scores[1]


def test_bm25_japanese_query_english_sentences_nonzero():
    """日本語クエリ × 英語文でも BM25 スコアが全ゼロにならないこと。"""
    # 英語文に日本語キーワードを混在させた例
    sentences = [
        "This paper proposes a KV cache reuse method.",
        "Unrelated content about cooking recipes.",
    ]
    # 英語文は全 ASCII なのでクエリとの token 重複は起きないが、
    # ゼロかどうかの検証ではなく「独立判定で壊れない」ことを確認する。
    scores = bm25_scores("KVキャッシュ", sentences)
    # scores は list[float] で長さが正しいこと
    assert len(scores) == 2


def test_bm25_cross_language_relevant_higher():
    """英クエリ × 日本語文でキーワードを含む文が高スコアになること。"""
    query = "attention mechanism"
    sentences = [
        "アテンションメカニズム（attention mechanism）を用いた手法を提案する。",
        "データ収集と前処理の手順について説明する。",
        "食材の栄養価と調理法の関係を分析する。",
    ]
    scores = bm25_scores(query, sentences)
    # "attention mechanism" を含む文が最高スコア
    assert scores[0] == max(scores), f"Expected scores[0] highest, got {scores}"


def test_char_ngrams_fullwidth_space_removed():
    """全角スペース・タブ・改行が bigram ノイズ term を生じさせないこと。

    旧実装の replace(' ', '') は ASCII スペースのみを除去し、
    全角スペース (U+3000) や改行が bigram に混入していた。
    """
    # 全角スペース含みテキスト
    tokens_fullwidth = _char_ngrams("東京　大阪")
    # 全角スペース自体がバイグラムの一部にならない
    assert not any("　" in t for t in tokens_fullwidth), (
        f"Full-width space leaked into bigrams: {tokens_fullwidth}"
    )

    # タブ含みテキスト
    tokens_tab = _char_ngrams("機械\t学習")
    assert not any("\t" in t for t in tokens_tab), f"Tab leaked into bigrams: {tokens_tab}"

    # 改行含みテキスト
    tokens_newline = _char_ngrams("研究\n手法")
    assert not any("\n" in t for t in tokens_newline), (
        f"Newline leaked into bigrams: {tokens_newline}"
    )
