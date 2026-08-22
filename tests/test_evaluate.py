import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from evaluate import _ARXIV_TO_PAPER_ID, evaluate, recall_at_k


def test_arxiv_map_has_all_ten_papers():
    assert len(_ARXIV_TO_PAPER_ID) == 10
    assert _ARXIV_TO_PAPER_ID["2608.07458"] == 10
    assert _ARXIV_TO_PAPER_ID["2410.10071"] == 1


def test_evaluate_accepts_arxiv_id(tmp_path):
    """evaluate() resolves arxiv_id → paper_id via _ARXIV_TO_PAPER_ID."""
    chunks_by_paper = {
        10: [{"text": "CoinRAG KV cache reuse", "nugget": "KV cache reuse"}]
    }
    gold = [{"arxiv_id": "2608.07458", "query": "KV cache", "answer_spans": ["KV cache"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0
    assert result["nugget"]["recall"] == 1.0


def test_evaluate_legacy_paper_id_still_works():
    """Legacy gold items with paper_id continue to work."""
    chunks_by_paper = {
        10: [{"text": "CoinRAG KV cache reuse", "nugget": "KV cache reuse"}]
    }
    gold = [{"paper_id": 10, "query": "KV cache", "answer_spans": ["KV cache"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0
