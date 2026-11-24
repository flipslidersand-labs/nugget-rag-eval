"""EvalConfig and evaluate()'s config / legacy-keyword handling (#392)."""

from __future__ import annotations

import dataclasses
import json
import sys

import pytest

from eval import evaluate as ev
from eval.evaluate import EvalConfig, build_chunks_index, evaluate
from nugget_rag.paper_registry import ARXIV_MAP

PID = ARXIV_MAP["2410.10071"]
# The answer sentence shares only "rule" with the query, so it ranks below the first sentence:
# one nugget per chunk misses it, five nuggets per chunk include it.
TEXT = (
    "The zebra crossing rule is stated here. "
    "Background number one is unrelated. "
    "The answer is quokka habitat rule. "
    "Background number two is unrelated. "
    "Background number three is unrelated. "
    "Background number four is unrelated."
)
CHUNKS = [{"paper_id": PID, "arxiv_id": "2410.10071", "chunk_indices": [0], "text": TEXT}]
GOLD = [
    {"arxiv_id": "2410.10071", "query": "zebra crossing rule", "answer_spans": ["quokka habitat"]}
]


@pytest.fixture
def by_paper():
    return build_chunks_index(CHUNKS)


def test_defaults_match_the_legacy_defaults():
    assert dataclasses.asdict(EvalConfig()) == {
        "top_k": 5,
        "match": "exact",
        "estimator": "words",
        "nugget_window": 0,
        "nuggets_per_chunk": 3,
        "embed_weight": 0.5,
    }


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        EvalConfig().top_k = 1  # type: ignore[misc]


def test_nuggets_per_chunk_changes_nugget_result(by_paper):
    one = evaluate(by_paper, GOLD, EvalConfig(nuggets_per_chunk=1))
    five = evaluate(by_paper, GOLD, EvalConfig(nuggets_per_chunk=5))
    assert one["nugget"]["recall"] == 0
    assert five["nugget"]["recall"] == 1.0
    assert one["nugget"]["avg_tokens"] < five["nugget"]["avg_tokens"]
    assert one["full_chunk"] == five["full_chunk"]


def test_default_config_equals_legacy_call_with_default_values(by_paper):
    def stable(res):  # latency is wall-clock
        return {k: v for k, v in res.items() if k != "latency_ms"}

    new = evaluate(by_paper, GOLD)
    legacy = evaluate(
        by_paper,
        GOLD,
        top_k=5,
        embed_weight=0.5,
        estimator="words",
        nugget_window=0,
        match="exact",
    )
    assert stable(new) == stable(evaluate(by_paper, GOLD, EvalConfig()))
    assert stable(new) == stable(legacy)
    assert "nuggets_per_chunk" not in new  # result JSON keeps its shape


def test_legacy_keywords_fold_into_config(by_paper):
    legacy = evaluate(by_paper, GOLD, top_k=1, estimator="chars", match="normalized")
    cfg = evaluate(by_paper, GOLD, EvalConfig(top_k=1, estimator="chars", match="normalized"))
    assert legacy["match"] == cfg["match"] == "normalized"
    assert legacy["nugget"] == cfg["nugget"]
    assert legacy["full_chunk"] == cfg["full_chunk"]


def test_config_plus_legacy_keyword_is_a_value_error(by_paper):
    with pytest.raises(ValueError, match="top_k"):
        evaluate(by_paper, GOLD, EvalConfig(), top_k=3)


@pytest.mark.parametrize(
    "kwargs, msg", [({"match": "fuzzy"}, "match"), ({"estimator": "bytes"}, "estimator")]
)
def test_invalid_config_values_raise(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        EvalConfig(**kwargs)


def test_cli_nuggets_per_chunk_is_passed_through(monkeypatch, tmp_path, capsys):
    c, g = tmp_path / "c.json", tmp_path / "g.json"
    c.write_text(json.dumps(CHUNKS), encoding="utf-8")
    g.write_text(json.dumps(GOLD), encoding="utf-8")
    recalls = {}
    for n in ("1", "5"):
        argv = ["ev", "--chunks", str(c), "--gold", str(g), "--nuggets-per-chunk", n]
        monkeypatch.setattr(sys, "argv", argv)
        ev.main()
        out = capsys.readouterr().out
        recalls[n] = json.loads(out[out.index("{") :])["nugget"]["recall"]
    assert recalls == {"1": 0, "5": 1.0}
