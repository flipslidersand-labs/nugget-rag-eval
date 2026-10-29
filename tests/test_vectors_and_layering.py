"""Tests for nugget_rag/vectors.py and the scorer -> embedder layering rule (#185)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nugget_rag import embedder, scorer, vectors

PKG = Path(vectors.__file__).parent


def test_cosine_similarity_known_values():
    v = [1.0, 2.0, 3.0]
    assert vectors.cosine_similarity(v, v) == pytest.approx(1.0)
    assert vectors.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert vectors.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert vectors.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector


def test_embed_scores_matches_per_sentence_cosine():
    q = [1.0, 0.0]
    sents = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert vectors.embed_scores(q, sents) == [vectors.cosine_similarity(q, s) for s in sents]
    assert vectors.embed_scores(q, []) == []


def test_embedder_reexports_the_same_functions():
    """Existing imports from nugget_rag.embedder keep working and point at vectors."""
    assert embedder.cosine_similarity is vectors.cosine_similarity
    assert embedder.embed_scores is vectors.embed_scores


def _imported_modules(path: Path) -> set[str]:
    """All module names imported anywhere in the file, including inside functions."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("module", ["scorer.py", "chunker.py", "vectors.py"])
def test_algorithm_layer_does_not_import_the_http_client(module):
    """Algorithm modules must not depend on embedder.py (the infrastructure layer),
    not even through a function-level import hidden from the module graph."""
    imports = _imported_modules(PKG / module)
    assert not {m for m in imports if m.startswith("nugget_rag.embedder")}, (
        f"{module} imports nugget_rag.embedder; use nugget_rag.vectors for vector math"
    )


def test_vectors_is_stdlib_only():
    imports = _imported_modules(PKG / "vectors.py")
    assert not {m for m in imports if m.startswith("nugget_rag")}


def test_top_nuggets_hybrid_path_still_works_with_precomputed_vectors():
    sents = ["alpha beta.", "gamma delta.", "epsilon zeta."]
    out = scorer.top_nuggets(
        "alpha",
        sents,
        top_k=1,
        embed_weight=1.0,
        query_vec=[1.0, 0.0],
        sent_vecs=[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
    )
    assert out == ["gamma delta."]  # picked by the embedding score alone
