"""Schema validation and paper indexing for chunks and gold files.

Pure functions shared by evaluate / check_regression / scripts. Imports nothing from
``eval.evaluate`` (the dependency goes the other way).
"""

from __future__ import annotations

from nugget_rag.paper_registry import resolve_paper_key

_REQUIRED_FIELDS = frozenset({"query", "answer_spans"})


def has_paper_id(item: dict) -> bool:
    """Return True if *item* carries a usable identifier.

    A truthy ``arxiv_id`` or a non-None ``paper_id`` counts. ``arxiv_id: ""``
    and ``paper_id: null`` do not (they would resolve to no key and silently
    give recall 0). ``paper_id: 0`` is a valid id.
    """
    return bool(item.get("arxiv_id")) or item.get("paper_id") is not None


def check_id_types(item: dict, label: str) -> None:
    """Raise ValueError if ``arxiv_id`` / ``paper_id`` have the wrong type.

    ``arxiv_id`` must be a str and ``paper_id`` an int (bool excluded); ``None``
    or absent is allowed (the other field may carry the id). Wrong types such as
    a list would otherwise crash ``resolve_paper_key`` with a TypeError, and a
    hashable one (``paper_id: "3"``) would silently never match.
    """
    arxiv_id = item.get("arxiv_id")
    if arxiv_id is not None and not isinstance(arxiv_id, str):
        raise ValueError(f"{label} 'arxiv_id' must be a string, got {type(arxiv_id).__name__!r}")
    paper_id = item.get("paper_id")
    if paper_id is not None and (isinstance(paper_id, bool) or not isinstance(paper_id, int)):
        raise ValueError(f"{label} 'paper_id' must be an int, got {type(paper_id).__name__!r}")


def validate_chunks(chunks: list[dict]) -> None:
    """Validate chunks list, raising ValueError on schema violations.

    Checks:
    - ``chunks`` is a list (not a dict or other type).
    - Each chunk has at least one of ``arxiv_id`` or ``paper_id``, and they are
      a str / an int respectively (see :func:`check_id_types`).
    - Each chunk has a ``text`` field that is a string with at least one
      non-whitespace character (blank-only and non-string values are rejected).

    Raises:
        ValueError: On the first violation, with a descriptive message listing
            the index and the specific problem.
    """
    if not isinstance(chunks, list):
        raise ValueError(f"chunks must be a JSON array (list), got {type(chunks).__name__!r}")
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunks[{i}] must be a dict, got {type(chunk).__name__!r}")
        check_id_types(chunk, f"chunks[{i}]")
        if not has_paper_id(chunk):
            raise ValueError(
                f"chunks[{i}] must have 'arxiv_id' or 'paper_id' (both absent or falsy)"
            )
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"chunks[{i}] missing or empty 'text' field (must be a non-blank string) "
                "— recall would silently drop to 0"
            )


def build_chunks_index(chunks: list[dict]) -> dict[str | int, list[dict]]:
    """Group a flat chunks list into a dict keyed by paper identifier.

    Uses :func:`resolve_paper_key` so chunks resolve to the same key as gold
    items sharing the same arxiv_id, regardless of which paper_id they carry.
    Assumes *chunks* has already been validated by :func:`validate_chunks`.
    """
    chunks_by_paper: dict[str | int, list[dict]] = {}
    for c in chunks:
        key = resolve_paper_key(c)
        chunks_by_paper.setdefault(key, []).append(c)
    return chunks_by_paper


def validate_gold(gold: list[dict]) -> None:
    """Validate gold set items, raising ValueError on schema violations.

    Checks:
    - ``gold`` is a non-empty list (an empty gold would evaluate nothing).
    - Each item has at least one of ``arxiv_id`` or ``paper_id``, and they are
      a str / an int respectively (see :func:`check_id_types`).
    - Each item has ``query`` and ``answer_spans``.
    - ``query`` is a string (type check) with at least one non-whitespace
      character; the two failures are reported with distinct messages.
    - ``answer_spans`` is a non-empty list of strings.

    Raises:
        ValueError: On the first item that fails validation, with a descriptive
            message listing the index and the specific problem.
    """
    if not isinstance(gold, list):
        raise ValueError(f"gold must be a JSON array (list), got {type(gold).__name__!r}")
    if not gold:
        raise ValueError("gold is empty — nothing to evaluate")
    for i, item in enumerate(gold):
        if not isinstance(item, dict):
            raise ValueError(f"gold[{i}] must be a dict, got {type(item).__name__!r}")
        missing = _REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(f"gold[{i}] missing required fields: {sorted(missing)}")

        check_id_types(item, f"gold[{i}]")
        if not has_paper_id(item):
            raise ValueError(f"gold[{i}] must have 'arxiv_id' or 'paper_id' (both absent or falsy)")

        query = item["query"]
        if not isinstance(query, str):
            raise ValueError(f"gold[{i}] 'query' must be a string, got {type(query).__name__!r}")
        if not query.strip():
            raise ValueError(f"gold[{i}] 'query' is blank (whitespace only); it must have text")

        spans = item["answer_spans"]
        if not isinstance(spans, list):
            raise ValueError(
                f"gold[{i}] 'answer_spans' must be a list, got {type(spans).__name__!r}"
            )
        if len(spans) == 0:
            raise ValueError(
                f"gold[{i}] 'answer_spans' is empty — recall would always be 0 (silent error)"
            )
        non_str = [s for s in spans if not isinstance(s, str)]
        if non_str:
            raise ValueError(
                f"gold[{i}] 'answer_spans' contains non-string elements: {non_str[:3]}"
            )
        if any(not s.strip() for s in spans):
            raise ValueError(
                f"gold[{i}] 'answer_spans' contains an empty/blank span — "
                "exact mode would always hit (inflated recall)"
            )


def check_gold_chunk_coverage(
    chunks_by_paper: dict[str, list[dict]], gold: list[dict]
) -> tuple[int, list[str]]:
    """Check gold keys against chunks_by_paper; return (affected query count, warning lines).

    Pure: nothing is printed. The warning (count plus up to 5 example keys) is for the
    caller to show, so a driver calling evaluate() many times can show it once.

    Raises:
        ValueError: If *gold* is non-empty and no gold key resolves to any chunks
            (every query would silently score recall 0).
    """
    missing = [key for item in gold if (key := resolve_paper_key(item)) not in chunks_by_paper]
    if not missing:
        return 0, []

    examples = sorted({str(k) for k in missing})[:5]
    if len(missing) == len(gold):
        raise ValueError(
            f"no gold paper key found in chunks — all {len(gold)} queries would score "
            f"recall 0 (wrong --chunks file or unregistered arxiv_id?); examples: {examples}"
        )
    warning = (
        f"[WARN] {len(missing)}/{len(gold)} gold queries have no chunks "
        f"(keys not in chunks data): {examples}"
    )
    return len(missing), [warning]
