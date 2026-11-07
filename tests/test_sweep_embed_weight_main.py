"""main() and cache-file paths of scripts/sweep_embed_weight.py (#246). tmp only, fake embedder."""

from __future__ import annotations

import json
import sys

import pytest

from nugget_rag import embedder
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
    }
]
GOLD = [{"arxiv_id": "2410.10071", "query": "alpha one", "answer_spans": ["Gamma three follows"]}]


class FakeClient:
    instances: list[FakeClient] = []

    def __init__(self, url):
        self.url, self.calls = url, []
        FakeClient.instances.append(self)

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


@pytest.fixture
def env(tmp_path, monkeypatch):
    FakeClient.instances = []
    monkeypatch.setattr(embedder, "EmbedClient", FakeClient)
    chunks, gold = tmp_path / "chunks.json", tmp_path / "gold.json"
    chunks.write_text(json.dumps(CHUNKS), encoding="utf-8")
    gold.write_text(json.dumps(GOLD), encoding="utf-8")
    return tmp_path, chunks, gold


def _run(monkeypatch, tmp_path, chunks, gold, *extra):
    argv = ["sw", "--chunks", str(chunks), "--gold", str(gold)]
    argv += ["--embedding-url", "http://embed.invalid", "--out-dir", str(tmp_path / "out"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    sw.main()


def test_main_writes_summary_and_cache(env, monkeypatch, capsys):
    tmp_path, chunks, gold = env
    _run(monkeypatch, tmp_path, chunks, gold, "--weights", "0.0", "0.5")
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"gold_w0.0", "gold_w0.5"}
    assert json.loads((tmp_path / "out" / "embed_cache.json").read_text(encoding="utf-8"))
    assert "gold | w |" in capsys.readouterr().out
    assert FakeClient.instances[0].url == "http://embed.invalid"


def test_main_with_two_gold_files_summarises_both(env, monkeypatch):
    tmp_path, chunks, gold = env
    gold2 = tmp_path / "gold_hard.json"
    gold2.write_text(json.dumps(GOLD), encoding="utf-8")
    argv = ["sw", "--chunks", str(chunks), "--gold", str(gold), str(gold2)]
    argv += ["--embedding-url", "http://embed.invalid", "--out-dir", str(tmp_path / "out")]
    argv += ["--weights", "0.5"]
    monkeypatch.setattr(sys, "argv", argv)
    sw.main()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"gold_w0.5", "gold_hard_w0.5"}


def test_main_resumes_from_existing_cache_without_calling_embed(env, monkeypatch):
    tmp_path, chunks, gold = env
    _run(monkeypatch, tmp_path, chunks, gold, "--weights", "0.5")
    _run(monkeypatch, tmp_path, chunks, gold, "--weights", "0.5")
    assert FakeClient.instances[0].calls
    assert FakeClient.instances[1].calls == []


@pytest.mark.parametrize("which", ["chunks", "gold"])
def test_main_exits_on_corrupt_json(env, monkeypatch, which):
    tmp_path, chunks, gold = env
    {"chunks": chunks, "gold": gold}[which].write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, chunks, gold)
    assert f"--{which} file is not valid JSON" in str(exc.value)
    assert not FakeClient.instances


def test_main_exits_on_schema_error_in_chunks(env, monkeypatch):
    tmp_path, chunks, gold = env
    chunks.write_text(json.dumps([{"paper_id": PID, "text": ""}]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, chunks, gold)
    assert str(exc.value).startswith("[ERROR] chunks[0]")


def test_main_exits_on_schema_error_in_gold(env, monkeypatch):
    tmp_path, chunks, gold = env
    gold.write_text(json.dumps([{"arxiv_id": "2410.10071", "query": "q"}]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, chunks, gold)
    assert str(exc.value).startswith("[ERROR] gold[0] missing required fields")


@pytest.mark.parametrize(
    "extra", [["--weights", "1.5"], ["--weights", "-0.1"], ["--top-k", "0"], ["--match", "fuzzy"]]
)
def test_main_rejects_bad_arguments(env, monkeypatch, extra, capsys):
    tmp_path, chunks, gold = env
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, chunks, gold, *extra)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err and extra[0] in err


def test_main_rejects_missing_input_file(env, monkeypatch, capsys):
    tmp_path, _, gold = env
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, tmp_path / "nope.json", gold)
    assert exc.value.code == 2
    assert "file not found" in capsys.readouterr().err


def test_load_cache_missing_file_or_none_gives_empty(tmp_path):
    assert sw.load_cache(None) == {}
    assert sw.load_cache(tmp_path / "absent.json") == {}


def test_main_exits_when_gold_and_chunks_do_not_match(env, monkeypatch):
    tmp_path, chunks, gold = env
    other = [{**CHUNKS[0], "paper_id": ARXIV_MAP["2508.11836"], "arxiv_id": "2508.11836"}]
    chunks.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, chunks, gold, "--weights", "0.0")
    assert str(exc.value.code).startswith("[ERROR] no gold paper key found in chunks")
