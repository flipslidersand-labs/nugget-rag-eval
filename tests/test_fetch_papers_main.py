"""In-process tests for scripts/fetch_papers.py main(): success path and the
failure_count / --max-failures gate (#176).

scripts/* is omitted from coverage (pyproject.toml), so the off-by-one-prone
threshold comparison is only protected by these tests.
"""

from __future__ import annotations

import json
import sys

import pytest

from scripts import fetch_papers as fp


def _chunk(pid: int, idx: int, text: str = "some text here") -> dict:
    return {"paper_id": pid, "chunk_index": idx, "text": text, "score": 1.0}


def _run(monkeypatch, tmp_path, *extra, papers=(1, 2), fail_pids=()):
    """Run main() with a fake API. Papers in *fail_pids* raise FetchError."""
    out = tmp_path / "out" / "chunks.json"
    monkeypatch.setattr(fp, "fetch_papers", lambda api: [{"id": p} for p in papers])

    def fake_fetch(api, pid, query, *a, **kw):
        if pid in fail_pids:
            raise fp.FetchError(f"boom {pid}")
        return [_chunk(pid, 0), _chunk(pid, 1)]

    monkeypatch.setattr(fp, "fetch_chunks_for_paper", fake_fetch)
    monkeypatch.setattr(sys, "argv", ["fetch_papers.py", "--out", str(out), *extra])
    return out


def test_success_path_writes_all_chunks_atomically(monkeypatch, tmp_path, capsys):
    out = _run(monkeypatch, tmp_path)
    fp.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [(c["paper_id"], c["chunk_index"]) for c in data] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert "Wrote 4 chunks" in capsys.readouterr().err
    assert not out.with_suffix(".tmp").exists()  # tmp file replaced, not left behind


def test_success_creates_missing_output_directory(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path)
    assert not out.parent.exists()
    fp.main()
    assert out.is_file()


def test_large_chunk_mode_combines_chunks(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, "--chunk-mode", "large", "--large-chunk-target", "100")
    fp.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 2  # one combined chunk per paper
    assert data[0]["chunk_indices"] == [0, 1]


def test_default_threshold_zero_any_failure_exits_1(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, fail_pids=(2,))
    with pytest.raises(SystemExit) as exc:
        fp.main()
    assert exc.value.code == 1
    assert "1 fetch failure(s) exceeded threshold (0)" in capsys.readouterr().err


def test_failures_equal_to_threshold_do_not_exit(monkeypatch, tmp_path):
    """Boundary: failure_count == --max-failures is allowed (strictly greater fails)."""
    out = _run(monkeypatch, tmp_path, "--max-failures", "1", fail_pids=(2,))
    fp.main()  # must not raise
    assert out.is_file()


def test_failures_one_over_threshold_exit_1(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, "--max-failures", "1", fail_pids=(1, 2))
    with pytest.raises(SystemExit) as exc:
        fp.main()
    assert exc.value.code == 1


def test_max_failures_zero_with_no_failures_passes(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, "--max-failures", "0")
    fp.main()
    assert out.is_file()


def test_partial_results_are_written_even_when_gate_fails(monkeypatch, tmp_path):
    """The output file is written before the gate, so partial data survives an exit 1."""
    out = _run(monkeypatch, tmp_path, fail_pids=(2,))
    with pytest.raises(SystemExit):
        fp.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {c["paper_id"] for c in data} == {1}


def test_fail_fast_aborts_before_writing(monkeypatch, tmp_path, capsys):
    out = _run(monkeypatch, tmp_path, "--fail-fast", fail_pids=(1,))
    with pytest.raises(SystemExit) as exc:
        fp.main()
    assert exc.value.code == 1
    assert "[ERROR]" in capsys.readouterr().err
    assert not out.exists()


def test_per_query_path_honours_failure_gate(monkeypatch, tmp_path):
    """--gold-set path: failure_count returned by fetch_per_query feeds the same gate."""
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps([{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["s"]}]),
        encoding="utf-8",
    )
    out = tmp_path / "chunks.json"
    monkeypatch.setattr(fp, "fetch_per_query", lambda *a, **kw: ([_chunk(1, 0)], 2))
    monkeypatch.setattr(
        sys, "argv", ["fetch_papers.py", "--gold-set", str(gold), "--out", str(out)]
    )
    with pytest.raises(SystemExit) as exc:
        fp.main()
    assert exc.value.code == 1
    assert json.loads(out.read_text(encoding="utf-8")) == [_chunk(1, 0)]

    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_papers.py", "--gold-set", str(gold), "--out", str(out), "--max-failures", "2"],
    )
    fp.main()  # 2 failures == threshold 2 -> allowed
