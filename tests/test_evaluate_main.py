"""Tests for evaluate.py — verbose path and main() CLI."""

from __future__ import annotations

import json
import sys

import pytest

from eval.evaluate import ARXIV_MAP, avg_tokens, evaluate

# ── verbose output path ──────────────────────────────────────────────────────


def _chunks_for(paper_id: int, text: str, nugget: str) -> dict[int, list[dict]]:
    return {paper_id: [{"text": text, "nugget": nugget}]}


def test_evaluate_verbose_prints_miss_to_stderr(capsys):
    """verbose=True with a nugget miss must print to stderr."""
    chunks_by_paper = {1: [{"text": "irrelevant", "nugget": "irrelevant"}]}
    gold = [{"paper_id": 1, "query": "missing span", "answer_spans": ["target answer"]}]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=True)
    captured = capsys.readouterr()
    assert "Nugget Recall Misses" in captured.err


def test_evaluate_verbose_no_miss_is_silent(capsys):
    """verbose=True with all nuggets hitting must not print to stderr."""
    chunks_by_paper = {1: [{"text": "target answer text", "nugget": "target answer"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["target answer"]}]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=True)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_evaluate_verbose_false_is_silent_on_miss(capsys):
    """verbose=False (default) must never print to stderr."""
    chunks_by_paper = {1: [{"text": "irrelevant", "nugget": "irrelevant"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["missing"]}]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=False)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_evaluate_verbose_caps_at_5_misses(capsys):
    """verbose output shows at most 5 misses even if more exist."""
    chunks_by_paper = {i: [{"text": "x", "nugget": "x"}] for i in range(1, 9)}
    gold = [{"paper_id": i, "query": f"q{i}", "answer_spans": ["miss"]} for i in range(1, 9)]
    evaluate(chunks_by_paper, gold, top_k=5, verbose=True)
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("  [")]
    assert len(lines) <= 5


# ── avg_tokens estimator ─────────────────────────────────────────────────────


def test_avg_tokens_chars_mode():
    results = [{"text": "abcd"}]  # 4 chars → 4/4 = 1.0
    assert avg_tokens(results, "text", estimator="chars") == 1.0


def test_avg_tokens_words_mode():
    results = [{"text": "one two three"}]
    assert avg_tokens(results, "text", estimator="words") == 3.0


def test_avg_tokens_empty_results():
    assert avg_tokens([], "text", estimator="words") == 0.0
    assert avg_tokens([], "text", estimator="chars") == 0.0


# ── evaluate() chars estimator ───────────────────────────────────────────────


def test_evaluate_chars_estimator_produces_result():
    chunks_by_paper = {1: [{"text": "eight chr", "nugget": "eight chr"}]}
    gold = [{"paper_id": 1, "query": "q", "answer_spans": ["eight chr"]}]
    result = evaluate(chunks_by_paper, gold, top_k=5, estimator="chars")
    assert result["full_chunk"]["avg_tokens"] > 0


def test_evaluate_empty_gold_returns_zeros():
    result = evaluate({}, [], top_k=5)
    assert result["n_queries"] == 0
    assert result["full_chunk"]["recall"] == 0
    assert result["nugget"]["recall"] == 0
    assert result["full_chunk"]["mrr"] == 0


# ── main() CLI via subprocess ─────────────────────────────────────────────────


def _write_json(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture()
def minimal_data(tmp_path):
    """Return (chunks_path, gold_path) with one passing entry."""
    paper_id = ARXIV_MAP["2410.10071"]  # == 1
    chunks = [
        {"paper_id": paper_id, "text": "unique answer phrase", "nugget": "unique answer phrase"}
    ]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["unique answer phrase"]}]
    c_path = tmp_path / "chunks.json"
    g_path = tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    return c_path, g_path


def test_main_bm25_only_exits_zero(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    main()  # must not raise
    out = capsys.readouterr().out
    assert "full-chunk" in out
    assert "nugget" in out


def test_main_chars_estimator(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--chunks",
            str(c_path),
            "--gold",
            str(g_path),
            "--token-estimator",
            "chars",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "chars" in out


def test_main_verbose_flag(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--verbose"],
    )
    main()
    out = capsys.readouterr().out
    assert "full-chunk" in out


def test_main_top_k_flag(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--top-k", "3"],
    )
    main()
    out = capsys.readouterr().out
    assert "Recall@3" in out


def test_main_outputs_json(minimal_data, monkeypatch, capsys):
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    main()
    out = capsys.readouterr().out
    # stdout ends with a JSON block starting at the first "{"
    json_part = out[out.find("{") :]
    parsed = json.loads(json_part)
    assert "full_chunk" in parsed
    assert "nugget" in parsed
    assert "n_queries" in parsed


def test_main_paper_id_chunk_key(tmp_path, monkeypatch, capsys):
    """Chunks with paper_id (int) are found by gold with arxiv_id resolved via ARXIV_MAP."""
    from eval.evaluate import main

    paper_id = ARXIV_MAP["2410.10071"]  # == 1
    chunks = [{"paper_id": paper_id, "text": "answer text", "nugget": "answer text"}]
    gold = [{"arxiv_id": "2410.10071", "query": "q", "answer_spans": ["answer text"]}]
    c_path, g_path = tmp_path / "chunks.json", tmp_path / "gold.json"
    _write_json(c_path, chunks)
    _write_json(g_path, gold)
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    main()
    out = capsys.readouterr().out
    json_part = out[out.find("{") :]
    parsed = json.loads(json_part)
    assert parsed["full_chunk"]["recall"] == 1.0


# ── CLI validation: --embed-weight / --top-k / --large-chunk-target ──────────


def test_main_invalid_embed_weight_above_1(minimal_data, monkeypatch):
    """--embed-weight > 1.0 must cause SystemExit (argparse error)."""
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--embed-weight", "2.5"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_main_invalid_embed_weight_negative(minimal_data, monkeypatch):
    """--embed-weight < 0.0 must cause SystemExit (argparse error)."""
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--embed-weight", "-1.0"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_main_embed_weight_boundary_values_accepted(minimal_data, monkeypatch, capsys):
    """--embed-weight 0.0 and 1.0 are valid boundary values."""
    from eval.evaluate import main

    c_path, g_path = minimal_data
    for weight in ("0.0", "1.0"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate.py",
                "--chunks",
                str(c_path),
                "--gold",
                str(g_path),
                "--embed-weight",
                weight,
            ],
        )
        main()  # must not raise


def test_main_invalid_top_k_zero(minimal_data, monkeypatch):
    """--top-k 0 must cause SystemExit (argparse error)."""
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--top-k", "0"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_main_invalid_top_k_negative(minimal_data, monkeypatch):
    """--top-k -1 must cause SystemExit (argparse error)."""
    from eval.evaluate import main

    c_path, g_path = minimal_data
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path), "--top-k", "-1"],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0


def test_float_between_0_1_valid():
    """_float_between_0_1 accepts values in [0, 1]."""
    from eval.evaluate import _float_between_0_1

    assert _float_between_0_1("0.0") == 0.0
    assert _float_between_0_1("0.5") == 0.5
    assert _float_between_0_1("1.0") == 1.0


def test_float_between_0_1_invalid():
    """_float_between_0_1 raises ArgumentTypeError for out-of-range values."""
    import argparse

    from eval.evaluate import _float_between_0_1

    with pytest.raises(argparse.ArgumentTypeError):
        _float_between_0_1("-0.1")
    with pytest.raises(argparse.ArgumentTypeError):
        _float_between_0_1("1.1")


def test_positive_int_valid():
    """_positive_int accepts values >= 1."""
    from eval.evaluate import _positive_int

    assert _positive_int("1") == 1
    assert _positive_int("10") == 10


def test_positive_int_invalid():
    """_positive_int raises ArgumentTypeError for zero or negative values."""
    import argparse

    from eval.evaluate import _positive_int

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-5")


# ── error handling: missing / invalid files (#103) ───────────────────────────


def test_main_missing_chunks_file_exits_with_message(tmp_path, monkeypatch, capsys):
    """--chunks pointing to a nonexistent file should exit via argparse, not a raw traceback."""
    from eval.evaluate import main

    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(tmp_path / "no_such.json"), "--gold", str(g_path)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "file not found" in err


def test_main_missing_gold_file_exits_with_message(tmp_path, monkeypatch, capsys):
    """--gold pointing to a nonexistent file should exit via argparse, not a raw traceback."""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    _write_json(c_path, [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--chunks", str(c_path), "--gold", str(tmp_path / "no_such.json")],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "file not found" in err


def test_main_invalid_json_chunks_exits_with_message(tmp_path, monkeypatch):
    """Malformed JSON in --chunks should produce a friendly [ERROR] message, not a traceback."""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    c_path.write_text("not valid json{{{")
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    assert "ERROR" in str(exc_info.value.code)


def test_main_non_utf8_chunks_exits_with_message(tmp_path, monkeypatch):
    """Non-UTF-8 bytes in --chunks should produce a friendly [ERROR] message, not a traceback."""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    c_path.write_bytes(b'[{"text": "\xff\xfe invalid utf-8"}]')
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "not valid UTF-8" in str(exc_info.value.code)


def test_main_invalid_json_gold_exits_with_message(tmp_path, monkeypatch):
    """Malformed JSON in --gold should produce a friendly [ERROR] message, not a traceback."""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    _write_json(c_path, [])
    g_path = tmp_path / "gold.json"
    g_path.write_text("{{invalid")
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
    assert "ERROR" in str(exc_info.value.code)


# ── validate_chunks in main() CLI (#149) ─────────────────────────────────────


def test_main_chunks_dict_instead_of_list_exits_with_error(tmp_path, monkeypatch):
    """chunks が JSON オブジェクト（dict）の場合は [ERROR] + SystemExit。"""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    c_path.write_text('{"paper_id": 1, "text": "x"}')
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "[ERROR]" in str(exc_info.value.code)


def test_main_chunks_missing_id_exits_with_error(tmp_path, monkeypatch):
    """ID フィールドが両方欠落した chunk は [ERROR] + SystemExit。"""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    _write_json(c_path, [{"text": "some text without id"}])
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "[ERROR]" in str(exc_info.value.code)


def test_main_chunks_missing_text_exits_with_error(tmp_path, monkeypatch):
    """text フィールドが欠落した chunk は [ERROR] + SystemExit。"""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    _write_json(c_path, [{"paper_id": 1}])
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "[ERROR]" in str(exc_info.value.code)


def test_main_chunks_empty_text_exits_with_error(tmp_path, monkeypatch):
    """text が空文字列の chunk は [ERROR] + SystemExit（サイレント recall 低下を防ぐ）。"""
    from eval.evaluate import main

    c_path = tmp_path / "chunks.json"
    _write_json(c_path, [{"paper_id": 1, "text": ""}])
    g_path = tmp_path / "gold.json"
    _write_json(g_path, [])
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--chunks", str(c_path), "--gold", str(g_path)]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "[ERROR]" in str(exc_info.value.code)
