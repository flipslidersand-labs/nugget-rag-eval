"""Tests for eval/check_regression.py — main() PASS / FAIL paths."""

from __future__ import annotations

import json
import subprocess
import sys

from eval.evaluate import ARXIV_MAP


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _passing_data(tmp_path):
    """Chunks and gold where both full-chunk and nugget recall = 1.0."""
    paper_id = ARXIV_MAP["2410.10071"]  # == 1
    chunks = [
        {"paper_id": paper_id, "text": "correct answer here", "nugget": "correct answer here"}
    ]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["correct answer here"]}]
    c_path, g_path = tmp_path / "chunks.json", tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    return c_path, g_path


def _failing_data(tmp_path):
    """Chunks and gold where recall = 0 (no answer match)."""
    chunks = [{"paper_id": 1, "text": "totally unrelated content", "nugget": "totally unrelated"}]
    gold = [{"paper_id": 99, "query": "q", "answer_spans": ["missing answer"]}]
    c_path, g_path = tmp_path / "chunks.json", tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    return c_path, g_path


# ── subprocess runner ─────────────────────────────────────────────────────────


def _run_main(args: list[str]) -> subprocess.CompletedProcess:
    """Run check_regression.py as a subprocess.

    Returns a CompletedProcess with .stdout, .stderr, and .returncode.
    Uses a fresh interpreter each call — zero shared module state.
    """
    import pathlib

    repo_root = str(pathlib.Path(__file__).parent.parent)
    return subprocess.run(
        [sys.executable, "-m", "eval.check_regression", *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )


# ── PASS tests ────────────────────────────────────────────────────────────────


def test_main_pass_exits_zero(tmp_path):
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.5"])
    assert result.returncode == 0
    assert "[PASS]" in result.stdout


def test_main_pass_prints_recall_values(tmp_path):
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.5"])
    assert "full-chunk" in result.stdout
    assert "nugget" in result.stdout
    assert "threshold" in result.stdout


def test_main_pass_custom_threshold(tmp_path):
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.1"])
    assert "[PASS]" in result.stdout
    assert "0.1" in result.stdout


def test_main_pass_custom_top_k(tmp_path):
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(
        ["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.5", "--top-k", "3"]
    )
    assert "[PASS]" in result.stdout
    assert "Recall@3" in result.stdout


# ── FAIL tests ────────────────────────────────────────────────────────────────


def test_main_fail_exits_one_on_low_recall(tmp_path):
    c_path, g_path = _failing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.95"])
    assert result.returncode == 1


def test_main_fail_prints_fail_message(tmp_path):
    c_path, g_path = _failing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.95"])
    assert "[FAIL]" in result.stdout
    assert "Recall regression" in result.stdout


def test_main_fail_mentions_both_failing_modes(tmp_path):
    c_path, g_path = _failing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.95"])
    assert "full-chunk" in result.stdout
    assert "nugget" in result.stdout


def test_main_fail_threshold_zero_still_passes(tmp_path):
    """threshold=0.0 means any recall (even 0.0) passes."""
    c_path, g_path = _failing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.0"])
    assert "[PASS]" in result.stdout


def test_main_default_threshold_is_0_95(tmp_path):
    """Running without --threshold uses 0.95 default and prints it."""
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path)])
    assert "0.95" in result.stdout


# ── arxiv_id-keyed chunks (PR #78 追加テスト) ────────────────────────────────


def test_arxiv_id_only_chunks_do_not_raise_key_error(tmp_path):
    """arxiv_id のみを持つチャンク（paper_id なし）でも KeyError が出ない。"""
    chunks = [{"arxiv_id": "2410.10071", "text": "answer text", "nugget": "answer text"}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["answer"]}]
    c_path, g_path = tmp_path / "c.json", tmp_path / "g.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "0.0"])
    assert "KeyError" not in result.stderr, "KeyError が発生した"


# ── --verbose flag ────────────────────────────────────────────────────────────


def test_verbose_on_fail_prints_nugget_miss_to_stderr(tmp_path):
    """--verbose + FAIL: 失敗クエリの情報が stderr に出力される。"""
    paper_id = ARXIV_MAP["2410.10071"]
    chunks = [{"paper_id": paper_id, "text": "unrelated content here", "nugget": "unrelated"}]
    gold = [{"arxiv_id": "2410.10071", "query": "KV cache", "answer_spans": ["KV cache"]}]
    c_path, g_path = tmp_path / "c.json", tmp_path / "g.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    result = _run_main(
        [
            "--chunks",
            str(c_path),
            "--gold",
            str(g_path),
            "--threshold",
            "0.95",
            "--verbose",
        ]
    )
    assert result.returncode == 1
    assert "Nugget Recall Misses" in result.stderr


def test_verbose_not_set_no_miss_output(tmp_path):
    """--verbose なし: stderr に Nugget Recall Misses が出ない。"""
    paper_id = ARXIV_MAP["2410.10071"]
    chunks = [{"paper_id": paper_id, "text": "unrelated", "nugget": "unrelated"}]
    gold = [{"arxiv_id": "2410.10071", "query": "KV cache", "answer_spans": ["KV cache"]}]
    c_path, g_path = tmp_path / "c.json", tmp_path / "g.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    result = _run_main(
        [
            "--chunks",
            str(c_path),
            "--gold",
            str(g_path),
            "--threshold",
            "0.95",
        ]
    )
    assert result.returncode == 1
    assert "Nugget Recall Misses" not in result.stderr


# ── error handling: missing / invalid files (#103) ────────────────────────────


def test_missing_chunks_file_exits_with_message(tmp_path):
    """--chunks に存在しないファイルを渡すと argparse 段階でエラーメッセージが出る。"""
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    result = _run_main(["--chunks", str(tmp_path / "no_such.json"), "--gold", str(g_path)])
    assert result.returncode != 0
    assert "file not found" in result.stderr


def test_missing_gold_file_exits_with_message(tmp_path):
    """--gold に存在しないファイルを渡すと argparse 段階でエラーメッセージが出る。"""
    c_path = tmp_path / "chunks.json"
    _write_json(c_path, [])
    result = _run_main(["--chunks", str(c_path), "--gold", str(tmp_path / "no_such.json")])
    assert result.returncode != 0
    assert "file not found" in result.stderr


def test_negative_threshold_rejected_at_argparse(tmp_path):
    """負の --threshold は argparse エラー (exit 2)。CI 常時 PASS を防ぐ。"""
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "-0.1"])
    assert result.returncode == 2
    assert "must be in [0, 1]" in result.stderr


def test_threshold_above_one_rejected_at_argparse(tmp_path):
    """1 超の --threshold は argparse エラー (exit 2)。常時 FAIL を防ぐ。"""
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--threshold", "1.1"])
    assert result.returncode == 2
    assert "must be in [0, 1]" in result.stderr


def test_top_k_zero_rejected_at_argparse(tmp_path):
    """--top-k 0 は argparse エラー (exit 2)。空 retrieval による偽 FAIL を防ぐ。"""
    c_path, g_path = _passing_data(tmp_path)
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path), "--top-k", "0"])
    assert result.returncode == 2
    assert "must be >= 1" in result.stderr


def test_invalid_gold_schema_prints_error_not_traceback(tmp_path):
    """gold スキーマ不正時は traceback でなく [ERROR] 1 行 + exit 1。"""
    c_path, _ = _passing_data(tmp_path)
    g_path = tmp_path / "bad_gold.json"
    _write_json(g_path, [{"query": "q"}])  # missing required fields
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path)])
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_json_chunks_exits_with_error_message(tmp_path):
    """不正 JSON の --chunks は [ERROR] メッセージを出して終了する。"""
    c_path = tmp_path / "chunks.json"
    c_path.write_text("{{not json")
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    result = _run_main(["--chunks", str(c_path), "--gold", str(g_path)])
    assert result.returncode != 0
    assert "ERROR" in result.stderr
