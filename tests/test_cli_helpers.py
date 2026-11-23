"""Tests for eval/cli_helpers.py and its adoption by fetch_papers (#175)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pytest

from eval.cli_helpers import (
    existing_file,
    float_between_0_1,
    load_json,
    non_negative_int,
    positive_int,
    write_json_atomic,
)


def test_positive_int_bounds():
    assert positive_int("1") == 1
    for bad in ("0", "-3"):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(bad)


def test_non_negative_int_bounds():
    assert non_negative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")


def test_float_between_0_1_bounds():
    assert float_between_0_1("0") == 0.0 and float_between_0_1("1") == 1.0
    for bad in ("-0.01", "1.01"):
        with pytest.raises(argparse.ArgumentTypeError):
            float_between_0_1(bad)


def test_existing_file(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("[]")
    assert existing_file(str(f)) == str(f)
    with pytest.raises(argparse.ArgumentTypeError):
        existing_file(str(tmp_path / "missing.json"))
    with pytest.raises(argparse.ArgumentTypeError):
        existing_file(str(tmp_path))  # a directory is not a file


def test_load_json_ok(tmp_path):
    f = tmp_path / "a.json"
    f.write_text('[{"x": 1}]', encoding="utf-8")
    assert load_json(str(f), "gold") == [{"x": 1}]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, "gold file not found"),
        (b"\xff\xfe\x00bad", "gold file is not valid UTF-8"),
        (b"not json", "gold file is not valid JSON"),
    ],
)
def test_load_json_errors_exit_with_message(tmp_path, content, expected):
    f = tmp_path / "x.json"
    if content is not None:
        f.write_bytes(content)
    with pytest.raises(SystemExit) as exc:
        load_json(str(f), "gold")
    assert expected in str(exc.value.code)


def test_fetch_papers_gold_set_invalid_utf8_has_no_traceback(tmp_path):
    """#175: fetch_papers used to leak a traceback on UnicodeDecodeError."""
    bad = tmp_path / "gold.json"
    bad.write_bytes(b"\xff\xfe\x00bad")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.fetch_papers", "--gold-set", str(bad)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "[ERROR]" in result.stderr
    assert "Traceback" not in result.stderr


def test_write_json_atomic_format_and_no_tmp_left(tmp_path):
    out = tmp_path / "x.json"
    write_json_atomic(out, {"k": "日本語"})
    assert out.read_text(encoding="utf-8") == '{\n  "k": "日本語"\n}'
    write_json_atomic(out, [1], indent=None, trailing_newline=True)
    assert out.read_text(encoding="utf-8") == "[1]\n"
    write_json_atomic(out, [1], indent=1)
    assert out.read_text(encoding="utf-8") == "[\n 1\n]"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_write_json_atomic_extension_not_clobbering_sibling(tmp_path):
    # with_suffix(".tmp") would turn "x.json.tmp" into "x.json.tmp" itself / collide with a sibling
    out = tmp_path / "x.json.tmp"
    sibling = tmp_path / "x.json"
    sibling.write_text("keep", encoding="utf-8")
    write_json_atomic(out, {"a": 1})
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}
    assert sibling.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json", "x.json.tmp"]


def test_write_json_atomic_unserializable_keeps_existing(tmp_path):
    out = tmp_path / "x.json"
    out.write_text("old", encoding="utf-8")
    with pytest.raises(TypeError):
        write_json_atomic(out, {"k": object()})
    assert out.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_write_json_atomic_replace_failure_keeps_existing_and_cleans_tmp(tmp_path, monkeypatch):
    out = tmp_path / "x.json"
    out.write_text("old", encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(out, {"new": 1})
    assert out.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
