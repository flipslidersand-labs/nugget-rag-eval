"""Tests for evaluate.py — mrr_at_k, evaluate() arxiv_id resolution."""
from eval.evaluate import ARXIV_MAP, evaluate, mrr_at_k


def _make_results(texts: list[str], field: str = "text") -> list[dict]:
    return [{field: t} for t in texts]


# ---- mrr_at_k basic cases ----

def test_mrr_hit_at_rank1():
    results = _make_results(["the answer is here", "something else", "unrelated"])
    assert mrr_at_k(results, ["answer"], "text") == 1.0


def test_mrr_hit_at_rank2():
    results = _make_results(["first result", "the answer is here", "unrelated"])
    rr = mrr_at_k(results, ["answer"], "text")
    assert abs(rr - 0.5) < 1e-9


def test_mrr_hit_at_rank3():
    results = _make_results(["first", "second", "the answer is here"])
    rr = mrr_at_k(results, ["answer"], "text")
    assert abs(rr - 1 / 3) < 1e-9


def test_mrr_no_hit_returns_zero():
    results = _make_results(["apple", "banana", "cherry"])
    assert mrr_at_k(results, ["answer"], "text") == 0.0


def test_mrr_empty_results():
    assert mrr_at_k([], ["answer"], "text") == 0.0


def test_mrr_case_insensitive():
    results = _make_results(["The Answer Is Here"])
    assert mrr_at_k(results, ["answer is"], "text") == 1.0


def test_mrr_multiple_spans_first_match_wins():
    # span2 hits at rank 1, span1 at rank 2 — RR should be 1.0
    results = _make_results(["span2 content", "span1 content"])
    assert mrr_at_k(results, ["span1", "span2"], "text") == 1.0


# ---- mrr_at_k with nugget field ----

def test_mrr_nugget_field_rank1():
    results = [{"nugget": "relevant span"}, {"nugget": "other"}]
    assert mrr_at_k(results, ["relevant"], "nugget") == 1.0


def test_mrr_nugget_field_rank2():
    results = [{"nugget": "unrelated"}, {"nugget": "relevant span"}]
    rr = mrr_at_k(results, ["relevant"], "nugget")
    assert abs(rr - 0.5) < 1e-9


# ---- rank ordering matters ----

def test_mrr_rank_ordering_affects_score():
    """Hitting at rank 1 vs rank 3 should give different MRR."""
    results_early = _make_results(["the answer", "b", "c"])
    results_late = _make_results(["a", "b", "the answer"])
    rr_early = mrr_at_k(results_early, ["answer"], "text")
    rr_late = mrr_at_k(results_late, ["answer"], "text")
    assert rr_early > rr_late
    assert abs(rr_early - 1.0) < 1e-9
    assert abs(rr_late - 1 / 3) < 1e-9


# ---- evaluate() arxiv_id → paper_id resolution (#39) ----

def test_evaluate_arxiv_id_resolves_via_arxiv_map():
    """evaluate() must resolve arxiv_id through ARXIV_MAP to find chunks."""
    paper_id = ARXIV_MAP["2608.07458"]  # == 10
    chunks_by_paper = {
        paper_id: [{"text": "KV cache reuse answer", "nugget": "KV cache reuse"}]
    }
    gold = [{"arxiv_id": "2608.07458", "query": "KV cache", "answer_spans": ["KV cache"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0
    assert result["nugget"]["recall"] == 1.0


def test_evaluate_arxiv_id_not_in_map_returns_zero():
    """Unknown arxiv_id falls back to str key — chunks not found → recall 0."""
    chunks_by_paper = {99: [{"text": "something", "nugget": "something"}]}
    gold = [{"arxiv_id": "9999.99999", "query": "q", "answer_spans": ["something"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 0.0


def test_evaluate_legacy_paper_id_still_works():
    """Gold items with paper_id (no arxiv_id) continue to work."""
    chunks_by_paper = {10: [{"text": "KV cache answer", "nugget": "KV cache"}]}
    gold = [{"paper_id": 10, "query": "KV cache", "answer_spans": ["KV cache"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0
