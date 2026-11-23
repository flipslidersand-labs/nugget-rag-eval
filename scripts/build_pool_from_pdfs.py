"""Build a query-independent chunk pool directly from local PDFs.

Why: per-query fetch depends on the search API (hybrid mode needs the embedding
service) and builds the pool from the gold queries themselves, so the pool is
biased toward easy questions. This script chunks every paper in full, so the
pool does not depend on any running service or on the queries.

Usage:
    python scripts/build_pool_from_pdfs.py --pdf-dir data/papers --out data/chunks_pdf.json
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
from pathlib import Path

from eval.cli_helpers import positive_int, write_json_atomic
from nugget_rag.paper_registry import ARXIV_MAP
from nugget_rag.schema import make_chunk

# Simpler than nugget_rag.chunker.split_sentences: split only after an ASCII terminator
# followed by whitespace, so decimals such as "0.5" stay intact (as in the chunker), but
# there is no CJK terminator handling and no abbreviation handling ("et al.", "Fig. 3").
# Pool chunks are cut at a ~300-word target, so exact sentence boundaries do not matter here.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
PDFTOTEXT_TIMEOUT_SEC = 120
# A paper's text is well under 1 MB; this only stops a crafted PDF from filling memory.
PDFTOTEXT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
PDFTOTEXT_MAX_STDERR_BYTES = 64 * 1024


def _drain(stream, limit: int, out: list[bytes], exceeded: threading.Event, kill=None) -> None:
    """Read *stream*, keeping at most *limit* bytes in *out*.

    Past *limit*, set *exceeded* and call *kill* (stdout: stop the child, stop reading).
    Without *kill* (stderr) keep draining and discarding so the child never blocks on a full pipe.
    """
    total = 0
    while chunk := stream.read(65536):
        room = limit - total
        total += len(chunk)
        if room > 0:
            out.append(chunk[:room])
        if total > limit:
            exceeded.set()
            if kill is not None:
                kill()
                return


def pdf_to_text(pdf: Path) -> str:
    """Extract text with pdftotext (reading order, no layout) and collapse whitespace.

    stdout is read with a cap of PDFTOTEXT_MAX_OUTPUT_BYTES; a crafted PDF that emits more
    gets the process killed and a RuntimeError (#305).
    """
    out_parts: list[bytes] = []
    err_parts: list[bytes] = []
    out_exceeded = threading.Event()
    try:
        with subprocess.Popen(
            ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            readers = [
                threading.Thread(
                    target=_drain,
                    args=(
                        proc.stdout,
                        PDFTOTEXT_MAX_OUTPUT_BYTES,
                        out_parts,
                        out_exceeded,
                        proc.kill,
                    ),
                ),
                threading.Thread(
                    target=_drain,
                    args=(proc.stderr, PDFTOTEXT_MAX_STDERR_BYTES, err_parts, threading.Event()),
                ),
            ]
            for t in readers:
                t.start()
            try:
                proc.wait(timeout=PDFTOTEXT_TIMEOUT_SEC)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.wait()
                raise RuntimeError(
                    f"pdftotext timed out after {PDFTOTEXT_TIMEOUT_SEC}s for {pdf.name}"
                ) from exc
            finally:
                for t in readers:
                    t.join()
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext not found (install poppler-utils)") from exc
    except OSError as exc:
        raise RuntimeError(f"pdftotext could not be run for {pdf.name}: {exc}") from exc
    if out_exceeded.is_set():
        raise RuntimeError(
            f"pdftotext output exceeds {PDFTOTEXT_MAX_OUTPUT_BYTES} bytes for {pdf.name}"
        )
    if proc.returncode != 0:
        stderr = b"".join(err_parts).decode("utf-8", errors="replace")
        raise RuntimeError(f"pdftotext failed for {pdf.name}: {stderr.strip()[:200]}")
    text = b"".join(out_parts).decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, target_words: int) -> list[str]:
    """Group consecutive sentences into chunks of about target_words words."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in _SENT_BOUNDARY.split(text):
        if not sent.strip():
            continue
        n = len(sent.split())
        if current and size + n > target_words:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(sent)
        size += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_pool(pdf_dir: Path, target_words: int) -> list[dict]:
    pool: list[dict] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        arxiv_id = pdf.stem
        paper_id = ARXIV_MAP.get(arxiv_id)
        if paper_id is None:
            print(f"  WARNING: {pdf.name} is not in ARXIV_MAP, skipping", file=sys.stderr)
            continue
        text = pdf_to_text(pdf)
        if not text:
            print(
                f"  WARNING: {pdf.name} has no extractable text (scanned?), skipping",
                file=sys.stderr,
            )
            continue
        pieces = chunk_text(text, target_words)
        for i, piece in enumerate(pieces):
            pool.append(
                make_chunk(
                    paper_id, piece, 0.0, chunk_indices=[i], arxiv_id=arxiv_id, arxiv_id_first=True
                )
            )
        print(f"  {arxiv_id} (paper {paper_id}): {len(pieces)} chunks", file=sys.stderr)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--out", type=Path, default=Path("data/chunks_pdf.json"))
    parser.add_argument("--target-words", type=positive_int, default=300)
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        sys.exit(f"[ERROR] pdf dir not found: {args.pdf_dir}")
    try:
        pool = build_pool(args.pdf_dir, args.target_words)
    except RuntimeError as exc:
        sys.exit(f"[ERROR] {exc}")
    if not pool:
        sys.exit(
            f"[ERROR] no chunks built from {args.pdf_dir} "
            "(every PDF was skipped: empty extracted text or not in ARXIV_MAP; see WARNINGs above)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, pool)
    print(f"Wrote {len(pool)} chunks to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
