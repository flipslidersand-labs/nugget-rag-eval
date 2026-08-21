from nugget_rag.scorer import bm25_scores, top_nuggets


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
