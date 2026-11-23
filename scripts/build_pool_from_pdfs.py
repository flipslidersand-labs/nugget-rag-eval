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


def pdf_to_text(pdf: Path) -> str:
    """Extract text with pdftotext (reading order, no layout) and collapse whitespace."""
    try:
        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
            capture_output=True,
            text=True,
            check=False,
            timeout=PDFTOTEXT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"pdftotext timed out after {PDFTOTEXT_TIMEOUT_SEC}s for {pdf.name}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext not found (install poppler-utils)") from exc
    except OSError as exc:
        raise RuntimeError(f"pdftotext could not be run for {pdf.name}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"pdftotext failed for {pdf.name}: {proc.stderr.strip()[:200]}")
    return re.sub(r"\s+", " ", proc.stdout).strip()


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
