"""Tests for scripts/build_pool_from_pdfs.py.

Covers: chunk_text (decimal safety, size grouping), pdf_to_text (pdftotext
failure), build_pool (chunk shape, unregistered PDF skipped), main() exit paths.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import build_pool_from_pdfs as mod
from scripts.build_pool_from_pdfs import build_pool, chunk_text, pdf_to_text


def test_chunk_text_keeps_decimals_intact():
    """'0.5' must not be split into '0.' and '5' (split_sentences does that)."""
    text = "We add noise with a probability of 0.5 at a random SNR. Next sentence follows."
    joined = " ".join(chunk_text(text, target_words=100))
    assert "probability of 0.5 at" in joined


def test_chunk_text_groups_sentences_up_to_target():
    sentences = [f"Sentence number {i} has exactly six words." for i in range(10)]
    chunks = chunk_text(" ".join(sentences), target_words=14)
    assert len(chunks) > 1
    # no chunk is cut mid-sentence and no text is lost
    assert " ".join(chunks) == " ".join(sentences)
    assert all(c.endswith(".") for c in chunks)


def test_chunk_text_single_long_sentence_is_one_chunk():
    chunks = chunk_text("word " * 50 + "end.", target_words=10)
    assert len(chunks) == 1


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", target_words=10) == []


class _FakePopen:
    """Stand-in for subprocess.Popen: canned stdout/stderr bytes, no real process."""

    last: _FakePopen

    def __init__(self, stdout=b"", stderr=b"", returncode=0, hang=False):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.hang = hang
        self.killed = False
        self.wait_timeouts: list = []
        _FakePopen.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.hang and timeout is not None and not self.killed:
            raise subprocess.TimeoutExpired(cmd="pdftotext", timeout=timeout)
        return self.returncode

    def kill(self):
        self.killed = True


def _popen(**kwargs):
    return patch(
        "scripts.build_pool_from_pdfs.subprocess.Popen",
        side_effect=lambda *a, **k: _FakePopen(**kwargs),
    )


def test_pdf_to_text_collapses_whitespace():
    with _popen(stdout=b"a  b\n\nc\td "):
        assert pdf_to_text(Path("x.pdf")) == "a b c d"


def test_pdf_to_text_raises_on_pdftotext_failure():
    with _popen(returncode=1, stderr=b"boom"):
        with pytest.raises(RuntimeError, match="pdftotext failed for x.pdf: boom"):
            pdf_to_text(Path("x.pdf"))


def test_pdf_to_text_converts_missing_executable():
    with patch("scripts.build_pool_from_pdfs.subprocess.Popen", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="install poppler-utils"):
            pdf_to_text(Path("x.pdf"))


def test_pdf_to_text_converts_timeout_and_kills_process():
    with _popen(hang=True):
        with pytest.raises(RuntimeError, match="timed out"):
            pdf_to_text(Path("x.pdf"))
    assert _FakePopen.last.killed


def test_pdf_to_text_passes_timeout():
    with _popen(stdout=b"a"):
        pdf_to_text(Path("x.pdf"))
    assert 120 in _FakePopen.last.wait_timeouts


def test_pdf_to_text_oversized_output_kills_process_and_raises():
    """A crafted PDF must not be able to fill memory with pdftotext output (#305)."""
    with (
        patch.object(mod, "PDFTOTEXT_MAX_OUTPUT_BYTES", 100),
        _popen(stdout=b"x" * 101),
    ):
        with pytest.raises(RuntimeError, match="output exceeds 100 bytes for x.pdf"):
            pdf_to_text(Path("x.pdf"))
    assert _FakePopen.last.killed


def test_pdf_to_text_output_at_limit_is_accepted():
    with (
        patch.object(mod, "PDFTOTEXT_MAX_OUTPUT_BYTES", 100),
        _popen(stdout=b"x" * 100),
    ):
        assert pdf_to_text(Path("x.pdf")) == "x" * 100
    assert not _FakePopen.last.killed


def test_pdf_to_text_huge_stderr_does_not_fail_or_kill():
    with _popen(returncode=1, stderr=b"e" * 1_000_000):
        with pytest.raises(RuntimeError, match="pdftotext failed for x.pdf: e{200}$"):
            pdf_to_text(Path("x.pdf"))
    assert not _FakePopen.last.killed


def test_build_pool_skips_empty_text_with_warning(tmp_path, capsys):
    (tmp_path / "2410.10071.pdf").write_bytes(b"")  # scanned: no text
    (tmp_path / "2410.10072.pdf").write_bytes(b"")
    texts = {"2410.10071": "", "2410.10072": "Alpha beta."}
    with (
        patch.dict(mod.ARXIV_MAP, {"2410.10072": 2}),
        patch.object(mod, "pdf_to_text", side_effect=lambda p: texts[p.stem]),
    ):
        pool = build_pool(tmp_path, target_words=10)
    assert {c["arxiv_id"] for c in pool} == {"2410.10072"}
    assert "no extractable text" in capsys.readouterr().err


def test_pdf_to_text_converts_other_oserror():
    with patch(
        "scripts.build_pool_from_pdfs.subprocess.Popen", side_effect=PermissionError("denied")
    ):
        with pytest.raises(RuntimeError, match="could not be run for x.pdf: denied"):
            pdf_to_text(Path("x.pdf"))


def test_main_exits_with_error_when_pdftotext_missing_end_to_end(monkeypatch, tmp_path):
    (tmp_path / "2410.10071.pdf").write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["build", "--pdf-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]
    )
    with patch("scripts.build_pool_from_pdfs.subprocess.Popen", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert str(exc.value.code).startswith("[ERROR]")
    assert "install poppler-utils" in str(exc.value.code)


def test_main_all_pdfs_empty_text_exits_with_error(monkeypatch, tmp_path):
    (tmp_path / "2410.10071.pdf").write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["build", "--pdf-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]
    )
    with patch.object(mod, "pdf_to_text", return_value=""):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    msg = str(exc.value.code)
    assert msg.startswith("[ERROR]") and "no chunks built" in msg and "empty extracted text" in msg


def test_main_partial_empty_text_exits_zero(monkeypatch, tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "2410.10071.pdf").write_bytes(b"")
    (pdfs / "2410.10072.pdf").write_bytes(b"")
    out = tmp_path / "pool.json"
    monkeypatch.setattr(sys, "argv", ["build", "--pdf-dir", str(pdfs), "--out", str(out)])
    texts = {"2410.10071": "", "2410.10072": "Alpha beta."}
    with (
        patch.dict(mod.ARXIV_MAP, {"2410.10072": 2}),
        patch.object(mod, "pdf_to_text", side_effect=lambda p: texts[p.stem]),
    ):
        mod.main()  # returns normally (exit 0)
    assert {c["arxiv_id"] for c in json.loads(out.read_text(encoding="utf-8"))} == {"2410.10072"}


def test_main_exits_with_error_on_pdf_to_text_failure(monkeypatch, tmp_path):
    (tmp_path / "2410.10071.pdf").write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["build", "--pdf-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]
    )
    err = RuntimeError("pdftotext not found (install poppler-utils)")
    with patch.object(mod, "pdf_to_text", side_effect=err):
        with pytest.raises(SystemExit) as exc:
            mod.main()
    assert str(exc.value.code).startswith("[ERROR]")
    assert "poppler-utils" in str(exc.value.code)


def test_build_pool_shape_and_skips_unregistered(tmp_path):
    (tmp_path / "2410.10071.pdf").write_bytes(b"")
    (tmp_path / "9999.99999.pdf").write_bytes(b"")  # not in ARXIV_MAP
    with patch.object(mod, "pdf_to_text", return_value="One. Two. Three."):
        pool = build_pool(tmp_path, target_words=1)
    assert pool, "registered pdf must produce chunks"
    assert {c["arxiv_id"] for c in pool} == {"2410.10071"}
    first = pool[0]
    assert first["paper_id"] == 1
    assert first["chunk_indices"] == [0]
    assert list(first) == ["paper_id", "arxiv_id", "chunk_indices", "text", "score"]


def test_main_missing_pdf_dir_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["build", "--pdf-dir", str(tmp_path / "nope"), "--out", str(tmp_path / "o.json")],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "pdf dir not found" in str(exc.value.code)


def test_main_empty_dir_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["build", "--pdf-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "no chunks built" in str(exc.value.code)


def test_main_writes_pool(monkeypatch, tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "2410.10071.pdf").write_bytes(b"")
    out = tmp_path / "out" / "pool.json"
    monkeypatch.setattr(sys, "argv", ["build", "--pdf-dir", str(pdfs), "--out", str(out)])
    with patch.object(mod, "pdf_to_text", return_value="Alpha beta. Gamma delta."):
        mod.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data and data[0]["arxiv_id"] == "2410.10071"
    assert not out.with_suffix(".tmp").exists()


def test_main_rejects_non_positive_target_words(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["build", "--pdf-dir", str(tmp_path), "--target-words", "0"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2
