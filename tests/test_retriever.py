from nugget_rag.retriever import retrieve_full_chunk, retrieve_nuggets

CHUNKS = [
    {"paper_id": 1, "chunk_index": 0, "text": "Diffusion models generate images by iterative denoising.", "score": 0.9},
    {"paper_id": 1, "chunk_index": 1, "text": "KV cache reuse reduces inference cost in transformer models.", "score": 0.8},
    {"paper_id": 1, "chunk_index": 2, "text": "BM25 is a classical term-matching retrieval algorithm.", "score": 0.7},
    {"paper_id": 1, "chunk_index": 3, "text": "RAG combines retrieval with language model generation.", "score": 0.6},
    {"paper_id": 1, "chunk_index": 4, "text": "Nugget extraction selects query-relevant spans.", "score": 0.5},
    {"paper_id": 1, "chunk_index": 5, "text": "Image segmentation uses pixel-level labels.", "score": 0.4},
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