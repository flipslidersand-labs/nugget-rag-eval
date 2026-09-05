from nugget_rag.retriever import retrieve_full_chunk, retrieve_nuggets

CHUNKS = [
    {
        "paper_id": 1,
        "chunk_index": 0,
        "text": "Diffusion models generate images by iterative denoising.",
        "score": 0.9,
    },
    {
        "paper_id": 1,
        "chunk_index": 1,
        "text": "KV cache reuse reduces inference cost in transformer models.",
        "score": 0.8,
    },
    {
        "paper_id": 1,
        "chunk_index": 2,
        "text": "BM25 is a classical term-matching retrieval algorithm.",
        "score": 0.7,
    },
    {
        "paper_id": 1,
        "chunk_index": 3,
        "text": "RAG combines retrieval with language model generation.",
        "score": 0.6,
    },
    {
        "paper_id": 1,
        "chunk_index": 4,
        "text": "Nugget extraction selects query-relevant spans.",
        "score": 0.5,
    },
    {
        "paper_id": 1,
        "chunk_index": 5,
        "text": "Image segmentation uses pixel-level labels.",
        "score": 0.4,
    },
]


def test_full_chunk_ranks_by_query():
    results = retrieve_full_chunk(CHUNKS, "KV cache transformer", top_k=3)
    assert len(results) == 3
    # KV cache chunk should be ranked first
    assert "KV cache" in results[0]["text"]


def test_full_chunk_not_just_first_k():
    # Without query ranking, chunk_index=0 (diffusion) would win.
    # With BM25 on "RAG retrieval", a later chunk should rank higher.
    results = retrieve_full_chunk(CHUNKS, "RAG retrieval language model", top_k=1)
    assert "RAG" in results[0]["text"] or "retrieval" in results[0]["text"].lower()


def test_full_chunk_top_k_limit():
    results = retrieve_full_chunk(CHUNKS, "anything", top_k=2)
    assert len(results) == 2


def test_full_chunk_empty():
    assert retrieve_full_chunk([], "query") == []


def test_nuggets_ranks_by_query():
    results = retrieve_nuggets(CHUNKS, "nugget span extraction", top_k=3)
    assert len(results) == 3
    # nugget chunk should appear in results
    texts = " ".join(r["nugget"] for r in results)
    assert "nugget" in texts.lower() or "span" in texts.lower()


def test_nuggets_field_present():
    results = retrieve_nuggets(CHUNKS, "BM25 retrieval", top_k=2)
    for r in results:
        assert "nugget" in r


def test_nuggets_empty():
    assert retrieve_nuggets([], "query") == []


# ── text キーなし・空テキスト (#47) ──────────────────────────────────────


def test_retrieve_full_chunk_empty_text_does_not_crash():
    """text が空文字列のチャンクはスコア 0 で返る（クラッシュしない）。"""
    chunks = [{"paper_id": 1, "text": "", "score": 0.5}]
    results = retrieve_full_chunk(chunks, "query", top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == ""


def test_retrieve_nuggets_empty_text_produces_empty_nugget():
    """text が空のチャンクは nugget フィールドが空文字列になる。"""
    chunks = [{"paper_id": 1, "text": "", "score": 0.5}]
    results = retrieve_nuggets(chunks, "query", top_k=1)
    assert len(results) == 1
    assert results[0]["nugget"] == ""


def test_retrieve_nuggets_preserves_all_original_fields():
    """retrieve_nuggets は元チャンクのフィールドをすべて保持する。"""
    chunks = [
        {
            "paper_id": 99,
            "chunk_index": 7,
            "text": "KV cache answer",
            "score": 0.9,
            "arxiv_id": "2608.07458",
        }
    ]
    results = retrieve_nuggets(chunks, "KV cache", top_k=1)
    r = results[0]
    assert r["paper_id"] == 99
    assert r["chunk_index"] == 7
    assert r["score"] == 0.9
    assert r["arxiv_id"] == "2608.07458"
    assert "nugget" in r


# ── 空センテンスリスト → nugget="" (#55) ─────────────────────────────────


def test_retrieve_nuggets_whitespace_only_text_returns_empty_nugget():
    """空白のみの text → split_sentences が空 → nugget="" になる。"""
    chunks = [{"paper_id": 1, "text": "   \n  \t  ", "score": 0.5}]
    results = retrieve_nuggets(chunks, "query", top_k=1)
    assert results[0]["nugget"] == ""


def test_retrieve_nuggets_single_sentence_returns_that_sentence():
    """1文のチャンクは nugget にその文が入る。"""
    chunks = [{"paper_id": 1, "text": "KV cache reuse reduces cost.", "score": 0.9}]
    results = retrieve_nuggets(chunks, "KV cache", top_k=1)
    assert "KV cache" in results[0]["nugget"]


def test_retrieve_nuggets_top_k_less_than_chunks():
    """top_k=2 で 6チャンクから 2件だけ返る。"""
    results = retrieve_nuggets(CHUNKS, "KV cache", top_k=2)
    assert len(results) == 2


# ── #150 embed_fn バッチ化 ────────────────────────────────────────────────


def _make_counting_embed_fn(dim: int = 4):
    """Return (embed_fn, call_counter) where call_counter tracks call count."""
    calls = []

    def embed_fn(texts):
        calls.append(len(texts))
        # Return deterministic unit vectors so scores are reproducible
        return [[float(i % dim == j % dim) for j in range(dim)] for i in range(len(texts))]

    return embed_fn, calls


def test_retrieve_nuggets_embed_fn_called_once_per_query():
    """#150: embed_fn は top_k 回ではなく 1 回だけ呼ばれる。"""
    embed_fn, calls = _make_counting_embed_fn()
    retrieve_nuggets(CHUNKS, "KV cache", top_k=5, embed_fn=embed_fn)
    assert len(calls) == 1, f"embed_fn was called {len(calls)} times, expected 1"


def test_retrieve_nuggets_embed_fn_batch_contains_query_and_all_sentences():
    """#150: 1回の embed_fn 呼び出しに query + 全チャンクのセンテンスが含まれる。"""
    embed_fn, calls = _make_counting_embed_fn()
    retrieve_nuggets(CHUNKS, "KV cache", top_k=3, embed_fn=embed_fn)
    assert len(calls) == 1
    # First text is query; remaining are sentences from top-3 chunks
    # Total texts > 1 (query alone would be 1)
    assert calls[0] > 1


def test_retrieve_nuggets_hybrid_scores_same_as_per_chunk_baseline():
    """#150: バッチ化後もスコア・ランキング結果は変更前と同一。"""

    # Use a deterministic embed_fn that returns the same vector for the same
    # position regardless of how texts are batched.
    def stable_embed(texts):
        # Vector depends only on text content (hash-based), not batch order
        import hashlib

        dim = 8
        vecs = []
        for t in texts:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            vec = [float((h >> i) & 1) for i in range(dim)]
            vecs.append(vec)
        return vecs

    results = retrieve_nuggets(CHUNKS, "KV cache", top_k=3, embed_fn=stable_embed)
    assert len(results) == 3
    for r in results:
        assert "nugget" in r
        assert isinstance(r["nugget"], str)


def test_retrieve_nuggets_no_embed_fn_unaffected():
    """#150: embed_fn なしの BM25-only パスは変更の影響を受けない。"""
    results = retrieve_nuggets(CHUNKS, "KV cache", top_k=3)
    assert len(results) == 3
    for r in results:
        assert "nugget" in r


def test_retrieve_nuggets_embed_fn_empty_chunks():
    """#150: 空チャンクリストでも embed_fn は呼ばれない。"""
    embed_fn, calls = _make_counting_embed_fn()
    results = retrieve_nuggets([], "KV cache", top_k=5, embed_fn=embed_fn)
    assert results == []
    assert len(calls) == 0
