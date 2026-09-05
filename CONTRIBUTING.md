# Contributing to nugget-rag-eval

## Prerequisites

- Python 3.12+
- `pip install -e ".[dev]"`

## Development workflow

1. Create a branch from `main` (`git checkout -b feat/issue-<N>-<slug>`)
2. Make changes
3. Run tests: `pytest tests/`
4. Run linter: `ruff check . && ruff format --check .`
5. Push and open a PR against `main`

## Project layout

```text
nugget_rag/          # core library — chunker, scorer, retriever, embedder, paper_registry
eval/                # evaluation scripts — evaluate.py, check_regression.py
scripts/             # data fetch utilities — fetch_papers.py
tests/               # pytest suite
tests/fixtures/      # minimal test data for CI (chunks_test.json, gold_test.json)
```

## Adding a new paper to ARXIV_MAP

Edit `nugget_rag/paper_registry.py` — one source of truth for arxiv_id → paper_id mapping.

## CI

| Job | Runner | Purpose |
|-----|--------|---------|
| `test` | `linux-general` (ARC) | lint + pytest with coverage ≥ 90% |
| `eval-regression` | `ubuntu-latest` | BM25 recall check against fixture data |

Both must pass before a PR can merge.

## Commit style

```text
<type>(<scope>): <summary>

Closes #<issue>
```

Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`.
