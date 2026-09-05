"""Guard: this package must stay runtime-dependency-free (#138)."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_runtime_dependencies_are_empty():
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["dependencies"] == []
