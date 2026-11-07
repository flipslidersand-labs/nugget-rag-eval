"""Tests for scripts/build_pool_from_pdfs.py.

Covers: chunk_text (decimal safety, size grouping), pdf_to_text (pdftotext
failure), build_pool (chunk shape, unregistered PDF skipped), main() exit paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_pdf_to_text_collapses_whitespace():
    proc = MagicMock(returncode=0, stdout="a  b\n\nc\td ", stderr="")
    with patch("scripts.build_pool_from_pdfs.subprocess.run", return_value=proc):
        assert pdf_to_text(Path("x.pdf")) == "a b c d"


def test_pdf_to_text_raises_on_pdftotext_failure():
    proc = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch("scripts.build_pool_from_pdfs.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="pdftotext failed for x.pdf"):
            pdf_to_text(Path("x.pdf"))


def test_pdf_to_text_converts_missing_executable():
    with patch("scripts.build_pool_from_pdfs.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="install poppler-utils"):
            pdf_to_text(Path("x.pdf"))


def test_pdf_to_text_converts_timeout():
    exc = subprocess.TimeoutExpired(cmd="pdftotext", timeout=120)
    with patch("scripts.build_pool_from_pdfs.subprocess.run", side_effect=exc):
        with pytest.raises(RuntimeError, match="timed out"):
            pdf_to_text(Path("x.pdf"))


def test_pdf_to_text_passes_timeout():
    proc = MagicMock(returncode=0, stdout="a", stderr="")
    with patch("scripts.build_pool_from_pdfs.subprocess.run", return_value=proc) as run:
        pdf_to_text(Path("x.pdf"))
    assert run.call_args.kwargs["timeout"] == 120


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
    assert set(first) == {"paper_id", "arxiv_id", "chunk_indices", "text", "score"}


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
