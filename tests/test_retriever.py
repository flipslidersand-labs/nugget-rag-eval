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
    chunks = [{"paper_id": 99, "chunk_index": 7, "text": "KV cache answer", "score": 0.9, "arxiv_id": "2608.07458"}]
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
