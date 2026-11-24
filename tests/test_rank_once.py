"""The BM25 chunk ranking runs once per query and is shared by the modes (#390)."""

from __future__ import annotations

from eval import evaluate as ev
from nugget_rag import retriever
from nugget_rag.paper_registry import ARXIV_MAP

PID = ARXIV_MAP["2410.10071"]
CHUNKS = {
    PID: [
        {"paper_id": PID, "chunk_index": 0, "text": "alpha beta gamma. delta epsilon zeta."},
        {"paper_id": PID, "chunk_index": 1, "text": "KV cache reuse cuts cost. Eta theta iota."},
        {"paper_id": PID, "chunk_index": 2, "text": "Kappa lambda mu. Cache eviction policy."},
    ]
}
GOLD = [
    {"arxiv_id": "2410.10071", "query": q, "answer_spans": [s]}
    for q, s in [("alpha", "alpha"), ("KV cache", "reuse"), ("eviction", "policy")]
]


def _count_chunk_rankings(monkeypatch) -> list[str]:
    """Record the query of every chunk-level bm25_scores call made by the retriever."""
    queries: list[str] = []
    real = retriever.bm25_scores

    def counting(query, texts):
        queries.append(query)
        return real(query, texts)

    monkeypatch.setattr(retriever, "bm25_scores", counting)
    return queries


def test_evaluate_ranks_chunks_once_per_query(monkeypatch):
    queries = _count_chunk_rankings(monkeypatch)
    ev.evaluate(CHUNKS, GOLD)
    assert queries == [g["query"] for g in GOLD]


def test_precomputed_ranking_skips_ranking_and_gives_same_results(monkeypatch):
    chunks = CHUNKS[PID]
    plain_full = retriever.retrieve_full_chunk(chunks, "KV cache", 2)
    plain_nug = retriever.retrieve_nuggets(chunks, "KV cache", 2)
    ranked = retriever.rank_chunks(chunks, "KV cache")
    queries = _count_chunk_rankings(monkeypatch)
    assert retriever.retrieve_full_chunk(chunks, "KV cache", 2, ranked=ranked) == plain_full
    assert retriever.retrieve_nuggets(chunks, "KV cache", 2, ranked=ranked) == plain_nug
    assert queries == []


def test_rank_chunks_of_nothing_is_empty():
    assert retriever.rank_chunks([], "q") == []


def test_latency_includes_the_shared_ranking_for_both_modes(monkeypatch):
    """The ranking time is added to each mode that uses it, so the modes stay comparable."""
    ticks = iter(range(1000))
    monkeypatch.setattr(ev.time, "perf_counter", lambda: next(ticks) / 1000)
    monkeypatch.setattr(ev, "rank_chunks", lambda chunks, query: list(chunks))
    res = ev.evaluate(CHUNKS, GOLD[:1])
    # ranking = 1 tick; each mode adds its own retrieval (1 tick) on top of it
    assert res["latency_ms"]["full_chunk"]["p50"] == res["latency_ms"]["nugget"]["p50"] == 2.0


def test_mode_without_ranking_support_keeps_the_four_argument_retrieve(monkeypatch):
    seen = []

    def retrieve(chunks, query, cfg, embed):
        seen.append(query)
        return chunks[:1]

    extra = ev.RetrievalMode("first", "text", "first", retrieve)
    monkeypatch.setattr(ev, "MODES", (*ev.MODES, extra))
    queries = _count_chunk_rankings(monkeypatch)
    ev.evaluate(CHUNKS, GOLD[:1])
    assert seen == ["alpha"]
    assert queries == ["alpha"]
