import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from evaluate import mrr_at_k


def test_mrr_first_hit():
    results = [{"text": "KV cache is key"}, {"text": "attention mechanism"}]
    assert mrr_at_k(results, ["KV cache"]) == 1.0


def test_mrr_second_hit():
    results = [{"text": "attention mechanism"}, {"text": "KV cache is key"}]
    assert mrr_at_k(results, ["KV cache"]) == 0.5


def test_mrr_no_hit():
    results = [{"text": "attention mechanism"}, {"text": "diffusion model"}]
    assert mrr_at_k(results, ["KV cache"]) == 0.0


def test_mrr_empty_results():
    assert mrr_at_k([], ["KV cache"]) == 0.0


def test_mrr_case_insensitive():
    results = [{"text": "KV CACHE reuse"}]
    assert mrr_at_k(results, ["kv cache"]) == 1.0


def test_mrr_custom_field():
    results = [{"nugget": "KV cache reuse"}, {"nugget": "other content"}]
    assert mrr_at_k(results, ["KV cache"], field="nugget") == 1.0
