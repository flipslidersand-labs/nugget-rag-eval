"""Tests for evaluate.py — mrr_at_k, evaluate() arxiv_id resolution, validate_gold."""

import pytest

from eval.evaluate import ARXIV_MAP, evaluate, mrr_at_k, recall_at_k, validate_gold
from tests.conftest import make_results

_make_results = make_results  # backward-compat alias for existing tests


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
    chunks_by_paper = {paper_id: [{"text": "KV cache reuse answer", "nugget": "KV cache reuse"}]}
    gold = [{"arxiv_id": "2608.07458", "query": "KV cache", "answer_spans": ["KV cache"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0
    assert result["nugget"]["recall"] == 1.0


def test_evaluate_arxiv_id_not_in_map_returns_zero():
    """Unknown arxiv_id falls back to str key — chunks not found → recall 0 for that query."""
    chunks_by_paper = {99: [{"text": "something", "nugget": "something"}]}
    gold = [
        {"arxiv_id": "9999.99999", "query": "q", "answer_spans": ["something"]},
        {"paper_id": 99, "query": "q2", "answer_spans": ["no match here"]},
    ]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 0.0


def test_evaluate_legacy_paper_id_still_works():
    """Gold items with paper_id (no arxiv_id) continue to work."""
    chunks_by_paper = {10: [{"text": "KV cache answer", "nugget": "KV cache"}]}
    gold = [{"paper_id": 10, "query": "KV cache", "answer_spans": ["KV cache"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0


# ── arxiv_id / paper_id 両存在時の優先度 (#54) ───────────────────────────


def test_evaluate_arxiv_id_takes_priority_over_paper_id():
    """gold に arxiv_id と paper_id が両存在する場合、arxiv_id が優先される。"""
    real_paper_id = ARXIV_MAP["2410.10071"]  # == 1
    # chunks は arxiv_id 解決先（paper_id=1）に置く
    chunks_by_paper = {
        real_paper_id: [{"text": "correct answer here", "nugget": "correct answer here"}],
        999: [{"text": "wrong chunk", "nugget": "wrong chunk"}],
    }
    # paper_id=999 だが arxiv_id=2410.10071 → ARXIV_MAP 経由で paper_id=1 を選ぶ
    gold = [
        {
            "arxiv_id": "2410.10071",
            "paper_id": 999,
            "query": "q",
            "answer_spans": ["correct answer"],
        }
    ]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0  # arxiv_id 優先で正解チャンクに当たる


def test_evaluate_empty_string_arxiv_id_falls_back_to_paper_id():
    """arxiv_id が空文字列（falsy）なら paper_id にフォールバック。"""
    chunks_by_paper = {42: [{"text": "answer here", "nugget": "answer here"}]}
    gold = [{"arxiv_id": "", "paper_id": 42, "query": "q", "answer_spans": ["answer here"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0


def test_evaluate_none_arxiv_id_falls_back_to_paper_id():
    """arxiv_id が None（falsy）なら paper_id にフォールバック。"""
    chunks_by_paper = {42: [{"text": "answer here", "nugget": "answer here"}]}
    gold = [{"arxiv_id": None, "paper_id": 42, "query": "q", "answer_spans": ["answer here"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert result["full_chunk"]["recall"] == 1.0


def test_evaluate_unknown_arxiv_id_returns_zero_recall():
    """ARXIV_MAP にない arxiv_id は str キーとして使われ chunks を見つけられない。"""
    chunks_by_paper = {1: [{"text": "answer", "nugget": "answer"}]}
    gold = [
        {"arxiv_id": "9999.99999", "query": "q", "answer_spans": ["answer"]},
        {"paper_id": 1, "query": "q2", "answer_spans": ["zzz no match"]},
    ]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    # 9999.99999 は ARXIV_MAP にないので str キーで探し、chunks は int キー → miss
    assert result["full_chunk"]["recall"] == 0.0


# ── gold key ↔ chunks クロス整合性チェック (#145) ────────────────────────────


def test_evaluate_warns_on_partially_missing_keys(capsys):
    """chunks に無い gold key があれば stderr に [WARN] と件数・例が出る。"""
    chunks_by_paper = {1: [{"text": "answer here", "nugget": "answer here"}]}
    gold = [
        {"paper_id": 1, "query": "q", "answer_spans": ["answer here"]},
        {"paper_id": 999, "query": "q2", "answer_spans": ["answer here"]},
    ]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    err = capsys.readouterr().err
    assert "[WARN]" in err
    assert "1/2" in err
    assert "999" in err
    assert result["n_queries_without_chunks"] == 1


def test_evaluate_raises_when_all_keys_missing():
    """gold の全 key が chunks に無ければ ValueError で fail-fast。"""
    chunks_by_paper = {1: [{"text": "answer", "nugget": "answer"}]}
    gold = [
        {"paper_id": 998, "query": "q", "answer_spans": ["answer"]},
        {"arxiv_id": "9999.99999", "query": "q2", "answer_spans": ["answer"]},
    ]
    with pytest.raises(ValueError, match="no gold paper key found in chunks"):
        evaluate(chunks_by_paper, gold, top_k=5)


def test_evaluate_no_warning_when_all_keys_present(capsys):
    """正常データでは警告なし・n_queries_without_chunks=0。"""
    chunks_by_paper = {1: [{"text": "answer here", "nugget": "answer here"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["answer here"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    assert "[WARN]" not in capsys.readouterr().err
    assert result["n_queries_without_chunks"] == 0


def test_evaluate_empty_gold_no_error():
    """gold が空でも ValueError にならない（既存挙動維持）。"""
    result = evaluate({}, [], top_k=5)
    assert result["n_queries"] == 0
    assert result["n_queries_without_chunks"] == 0


def test_evaluate_warn_examples_capped_at_five(capsys):
    """未解決 key の例は最大5件まで。"""
    chunks_by_paper = {1: [{"text": "answer here", "nugget": "answer here"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["answer here"]}] + [
        {"paper_id": 900 + i, "query": f"q{i}", "answer_spans": ["x"]} for i in range(7)
    ]
    result = evaluate(chunks_by_paper, gold, top_k=5)
    err = capsys.readouterr().err
    assert "7/8" in err
    assert err.count("90") <= 5  # 900..906 のうち例示は5件まで
    assert result["n_queries_without_chunks"] == 7


# ── validate_gold (#108) ──────────────────────────────────────────────────────


def _valid_item(**overrides) -> dict:
    base = {"paper_id": 1, "query": "q", "answer_spans": ["span"]}
    base.update(overrides)
    return base


def test_validate_gold_passes_on_valid_items():
    gold = [_valid_item(), _valid_item(paper_id=2, answer_spans=["a", "b"])]
    validate_gold(gold)  # must not raise


def test_validate_gold_passes_on_empty_list():
    validate_gold([])  # empty gold is valid (evaluate() returns zeros)


def test_validate_gold_missing_query_raises():
    item = {"paper_id": 1, "answer_spans": ["span"]}
    with pytest.raises(ValueError, match="missing required fields"):
        validate_gold([item])


def test_validate_gold_missing_answer_spans_raises():
    item = {"paper_id": 1, "query": "q"}
    with pytest.raises(ValueError, match="missing required fields"):
        validate_gold([item])


def test_validate_gold_missing_both_id_fields_raises():
    item = {"query": "q", "answer_spans": ["span"]}
    with pytest.raises(ValueError, match="'arxiv_id' or 'paper_id'"):
        validate_gold([item])


def test_validate_gold_empty_answer_spans_raises():
    item = _valid_item(answer_spans=[])
    with pytest.raises(ValueError, match="'answer_spans' is empty"):
        validate_gold([item])


def test_validate_gold_answer_spans_not_list_raises():
    item = _valid_item(answer_spans="span")
    with pytest.raises(ValueError, match="must be a list"):
        validate_gold([item])


def test_validate_gold_answer_spans_non_string_element_raises():
    item = _valid_item(answer_spans=[42, "ok"])
    with pytest.raises(ValueError, match="non-string elements"):
        validate_gold([item])


def test_validate_gold_reports_correct_index():
    gold = [_valid_item(), _valid_item(answer_spans=[])]
    with pytest.raises(ValueError, match=r"gold\[1\]"):
        validate_gold(gold)


def test_validate_gold_arxiv_id_only_is_valid():
    item = {"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["span"]}
    validate_gold([item])  # must not raise


def test_validate_gold_both_ids_present_is_valid():
    item = {"arxiv_id": "2410.10071", "paper_id": 1, "query": "q", "answer_spans": ["span"]}
    validate_gold([item])  # must not raise


def test_evaluate_raises_on_empty_answer_spans():
    """evaluate() must propagate validate_gold's ValueError on empty answer_spans."""
    chunks_by_paper = {1: [{"text": "x", "nugget": "x"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": []}]
    with pytest.raises(ValueError, match="'answer_spans' is empty"):
        evaluate(chunks_by_paper, gold)


def test_evaluate_raises_on_missing_query():
    """evaluate() must propagate validate_gold's ValueError on missing query."""
    chunks_by_paper = {1: [{"text": "x", "nugget": "x"}]}
    gold = [{"paper_id": 1, "answer_spans": ["span"]}]
    with pytest.raises(ValueError, match="missing required fields"):
        evaluate(chunks_by_paper, gold)


# ---- recall_at_k per-result judgment (#146) ----


def test_recall_hit_within_single_result():
    results = make_results(["the answer span is here", "other text"])
    assert recall_at_k(results, ["answer span"]) is True


def test_recall_no_hit_returns_false():
    results = make_results(["nothing relevant", "still nothing"])
    assert recall_at_k(results, ["answer span"]) is False


def test_recall_empty_results():
    assert recall_at_k([], ["answer span"]) is False


def test_recall_case_insensitive():
    results = make_results(["The ANSWER Span"])
    assert recall_at_k(results, ["answer span"]) is True


def test_recall_nugget_field():
    results = make_results(["answer span here"], field="nugget")
    assert recall_at_k(results, ["answer span"], field="nugget") is True


def test_recall_no_cross_boundary_false_positive():
    """Span matching only across the join of two adjacent results must not count (#146)."""
    results = make_results(["tail of chunk answer", "span head of next chunk"])
    # "answer span" only appears if results are joined with a space
    assert "answer span" in " ".join(r["text"] for r in results)
    assert recall_at_k(results, ["answer span"]) is False


def test_recall_matches_mrr_positivity():
    """Invariant: recall_at_k(...) == (mrr_at_k(...) > 0) on the same inputs."""
    cases = [
        make_results(["the answer span is here"]),
        make_results(["miss", "answer span"]),
        make_results(["tail of chunk answer", "span head of next chunk"]),
        make_results(["no hit at all"]),
        [],
    ]
    for results in cases:
        assert recall_at_k(results, ["answer span"]) == (mrr_at_k(results, ["answer span"]) > 0)
