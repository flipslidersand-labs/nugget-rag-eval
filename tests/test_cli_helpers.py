"""Tests for eval/cli_helpers.py and its adoption by fetch_papers (#175)."""

from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from eval.cli_helpers import (
    existing_file,
    float_between_0_1,
    load_json,
    non_negative_int,
    positive_int,
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
