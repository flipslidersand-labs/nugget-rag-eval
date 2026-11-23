"""Shared chunks/gold loading used by the CLIs (#340)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.evaluate import load_chunks_index, load_gold, load_inputs

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = [{"paper_id": 1, "chunk_index": 0, "text": "alpha beta", "score": 0.9}]
GOLD = [{"arxiv_id": "2410.10071", "query": "alpha", "answer_spans": ["alpha"]}]
BAD_GOLD = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": []}]
GOLD_ERROR = "[ERROR] gold[0] 'answer_spans' is empty — recall would always be 0 (silent error)"
CHUNKS_ERROR = "[ERROR] chunks must be a JSON array (list), got 'dict'"


def _write(tmp_path, name, data, raw=False):
    p = tmp_path / name
    p.write_text(data if raw else json.dumps(data), encoding="utf-8")
    return str(p)


def test_load_inputs_returns_index_and_gold(tmp_path):
    index, gold = load_inputs(_write(tmp_path, "c.json", CHUNKS), _write(tmp_path, "g.json", GOLD))
    assert index == {1: CHUNKS}
    assert gold == GOLD


def test_load_inputs_chunks_schema_error_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_inputs(_write(tmp_path, "c.json", {"a": 1}), _write(tmp_path, "g.json", GOLD))
    assert exc.value.code == CHUNKS_ERROR


def test_load_inputs_gold_schema_error_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_inputs(_write(tmp_path, "c.json", CHUNKS), _write(tmp_path, "g.json", BAD_GOLD))
    assert exc.value.code == GOLD_ERROR


@pytest.mark.parametrize("which", ["chunks", "gold"])
def test_load_inputs_invalid_json_names_the_option(tmp_path, which):
    c = _write(tmp_path, "c.json", "[oops" if which == "chunks" else CHUNKS, raw=which == "chunks")
    g = _write(tmp_path, "g.json", "[oops" if which == "gold" else GOLD, raw=which == "gold")
    with pytest.raises(SystemExit) as exc:
        load_inputs(c, g)
    assert str(exc.value.code).startswith(f"[ERROR] --{which} file is not valid JSON (")


def test_load_inputs_reads_both_files_before_validating(tmp_path):
    """Chunks schema error + unreadable gold JSON: the JSON error wins (as evaluate always did)."""
    c = _write(tmp_path, "c.json", {"a": 1})
    g = _write(tmp_path, "g.json", "[oops", raw=True)
    with pytest.raises(SystemExit) as exc:
        load_inputs(c, g)
    assert str(exc.value.code).startswith("[ERROR] --gold file is not valid JSON")


def test_load_inputs_missing_file_exits(tmp_path):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(SystemExit) as exc:
        load_inputs(missing, _write(tmp_path, "g.json", GOLD))
    assert exc.value.code == f"[ERROR] --chunks file not found: {missing}"


def test_load_chunks_index_prefix_and_label(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_chunks_index(_write(tmp_path, "c.json", {"a": 1}), "--pool P", error_prefix="pool=P: ")
    assert exc.value.code == "[ERROR] pool=P: chunks must be a JSON array (list), got 'dict'"
    missing = str(tmp_path / "nope.json")
    with pytest.raises(SystemExit) as exc:
        load_chunks_index(missing, "--pool P", error_prefix="pool=P: ")
    assert exc.value.code == f"[ERROR] --pool P file not found: {missing}"


def test_load_chunks_index_ok(tmp_path):
    assert load_chunks_index(_write(tmp_path, "c.json", CHUNKS), "--chunks") == {1: CHUNKS}


def test_load_gold_prefix_and_ok(tmp_path):
    assert load_gold(_write(tmp_path, "g.json", GOLD), "--gold") == GOLD
    with pytest.raises(SystemExit) as exc:
        load_gold(_write(tmp_path, "b.json", BAD_GOLD), "--gold G", error_prefix="gold=G: ")
    assert exc.value.code == GOLD_ERROR.replace("[ERROR] ", "[ERROR] gold=G: ")


def _cli(tmp_path, argv):
    r = subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, cwd=ROOT, env=_env()
    )
    return r.returncode, r.stdout, r.stderr


def _env():
    import os

    return {**os.environ, "PYTHONPATH": str(ROOT)}


@pytest.mark.parametrize(
    "script",
    [
        ["-m", "eval.evaluate"],
        ["-m", "eval.check_regression"],
        ["scripts/analyze_nugget_failures.py"],
    ],
    ids=["evaluate", "check_regression", "analyze"],
)
@pytest.mark.parametrize(
    ("chunks", "gold", "message"),
    [
        ({"a": 1}, GOLD, CHUNKS_ERROR),
        (CHUNKS, BAD_GOLD, GOLD_ERROR),
    ],
    ids=["chunks-schema", "gold-schema"],
)
def test_clis_share_validation_errors_and_exit_code(tmp_path, script, chunks, gold, message):
    args = [
        "--chunks",
        _write(tmp_path, "c.json", chunks),
        "--gold",
        _write(tmp_path, "g.json", gold),
    ]
    if script[-1] == "eval.check_regression":
        args += ["--baseline", _write(tmp_path, "b.json", {})]
    code, out, err = _cli(tmp_path, [*script, *args])
    assert code == 1
    assert err.strip() == message
    assert out == ""
