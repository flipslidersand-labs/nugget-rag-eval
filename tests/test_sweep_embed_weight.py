"""Tests for scripts/sweep_embed_weight.py (#199): embed once, resume, sweep from cache."""

from __future__ import annotations

import json

import pytest

from eval.evaluate import build_chunks_index
from nugget_rag.paper_registry import ARXIV_MAP
from scripts import sweep_embed_weight as sw

PID = ARXIV_MAP["2410.10071"]
CHUNKS = [
    {
        "paper_id": PID,
        "arxiv_id": "2410.10071",
        "chunk_indices": [0],
        "text": "Alpha one is here. Beta two is there. Gamma three follows. Delta four ends.",
        "score": 1.0,
    },
    {
        "paper_id": PID,
        "arxiv_id": "2410.10071",
        "chunk_indices": [1],
        "text": "Alpha one is here. Epsilon five appears. Zeta six closes.",
        "score": 1.0,
    },
]
GOLD = [
    {"arxiv_id": "2410.10071", "query": "alpha one", "answer_spans": ["Gamma three follows"]},
    {"arxiv_id": "2410.10071", "query": "epsilon five", "answer_spans": ["Zeta six closes"]},
]


def _fake_embedder():
    calls: list[list[str]] = []

    def embed(texts):
        calls.append(list(texts))
        # deterministic 2-d vector from the text, so cosine differs per sentence
        return [[float(len(t) % 7), float(sum(map(ord, t)) % 11)] for t in texts]

    return embed, calls


def test_collect_texts_matches_what_evaluate_will_request():
    by = build_chunks_index(CHUNKS)
    texts = sw.collect_texts(by, GOLD, top_k=5, match="exact")
    assert "alpha one" in texts and "epsilon five" in texts  # queries are embedded
    assert "Zeta six closes." in texts
    assert len(texts) == len(set(texts))  # unique


def test_each_unique_text_is_embedded_exactly_once_across_all_weights():
    by = build_chunks_index(CHUNKS)
    texts = sw.collect_texts(by, GOLD, top_k=5, match="exact")
    embed, calls = _fake_embedder()
    cache: dict = {}
    n = sw.warm_cache(texts, embed, cache, log=lambda *_: None)
    assert n == len(texts)
    embedded = [t for batch in calls for t in batch]
    assert sorted(embedded) == sorted(texts)  # once each, no repeats

    weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    before = len(calls)
    res = sw.sweep(by, {"g": GOLD}, weights, sw.cached_embed_fn(cache), match="exact")
    assert len(calls) == before  # the sweep never touches the network
    assert set(res) == {f"g_w{w}" for w in weights}


def test_resume_embeds_only_what_is_missing(tmp_path):
    by = build_chunks_index(CHUNKS)
    texts = sw.collect_texts(by, GOLD, top_k=5, match="exact")
    cache_path = tmp_path / "cache.json"
    embed, calls = _fake_embedder()
    cache = {t: [1.0, 1.0] for t in texts[:3]}  # pretend an earlier run finished 3
    sw.save_cache(cache_path, cache)

    loaded = sw.load_cache(cache_path)
    n = sw.warm_cache(texts, embed, loaded, cache_path, log=lambda *_: None)
    assert n == len(texts) - 3
    assert {t for b in calls for t in b} == set(texts[3:])
    assert set(json.loads(cache_path.read_text())) == set(texts)  # persisted
    assert not cache_path.with_suffix(".tmp").exists()


def test_warm_cache_batches_and_reports_progress():
    texts = [f"t{i}" for i in range(300)]
    embed, calls = _fake_embedder()
    logs: list[str] = []
    sw.warm_cache(texts, embed, {}, log=logs.append)
    assert [len(b) for b in calls] == [128, 128, 44]
    assert any("300/300" in line for line in logs)


def test_warm_cache_rejects_a_short_response():
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2"):
        sw.warm_cache(["a", "b"], lambda ts: [[0.0]], {}, log=lambda *_: None)


def test_cached_embed_fn_fails_loudly_on_a_missing_text():
    fn = sw.cached_embed_fn({"a": [1.0]})
    assert fn(["a"]) == [[1.0]]
    with pytest.raises(KeyError, match="not pre-embedded"):
        fn(["a", "b"])


def test_embed_weight_actually_changes_the_nugget_that_is_chosen():
    """The answer sentence shares no words with the query: BM25 (w=0) cannot find it among
    the 3 kept sentences, the embedding (w=1) can. The sweep must show that difference."""
    chunk = {
        "paper_id": PID,
        "arxiv_id": "2410.10071",
        "chunk_indices": [0],
        "text": "Alpha one is here. Bb cc. Dd ee. Ff gg. Hh ii. Zeta six closes.",
        "score": 1.0,
    }
    gold = [{"arxiv_id": "2410.10071", "query": "alpha one", "answer_spans": ["Zeta six closes"]}]
    by = build_chunks_index([chunk])
    texts = sw.collect_texts(by, gold, top_k=5, match="exact")
    # query and the answer sentence point the same way; every other sentence is orthogonal
    cache = {t: [1.0, 0.0] if t in ("alpha one", "Zeta six closes.") else [0.0, 1.0] for t in texts}
    res = sw.sweep(by, {"g": gold}, [0.0, 1.0], sw.cached_embed_fn(cache), match="exact")
    assert res["g_w0.0"]["nugget"]["recall"] == 0.0
    assert res["g_w1.0"]["nugget"]["recall"] == 1.0
    assert res["g_w0.0"]["full_chunk"] == res["g_w1.0"]["full_chunk"]  # full-chunk ignores w


def test_format_table_lists_every_run():
    by = build_chunks_index(CHUNKS)
    texts = sw.collect_texts(by, GOLD, top_k=5, match="exact")
    cache = {t: [1.0, 0.5] for t in texts}
    res = sw.sweep(by, {"g": GOLD}, [0.0, 0.5], sw.cached_embed_fn(cache), match="exact")
    table = sw.format_table(res)
    assert "g | 0.0 |" in table and "g | 0.5 |" in table
