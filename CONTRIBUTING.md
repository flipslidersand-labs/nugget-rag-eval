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
nugget_rag/          # core library (chunker.py, scorer.py, retriever.py, etc.)
eval/                # evaluation CLIs (evaluate.py, check_regression.py, etc.), gold sets, and
                     #   shared argparse helpers (cli_helpers.py, also imported by scripts/)
scripts/             # data fetch / pool build / analysis scripts
tests/               # pytest suite
tests/fixtures/      # minimal test data for CI (chunks_test.json, gold_test.json)
docs/adr/            # architecture decision records
```

Run `git ls-files` for the per-file list; it is not enumerated here to avoid drift.

## Adding a new paper to ARXIV_MAP

Edit `nugget_rag/paper_registry.py` — one source of truth for arxiv_id → paper_id mapping.

## CI

| Job | Purpose |
|-----|---------|
| `markdownlint` | markdownlint-cli with `.markdownlint.json` over all `*.md` |
| `test` | reusable workflow (`qa-workflows/python-test.yml`); coverage threshold 90% (`coverage-threshold: 90` in ci.yml; `fail_under` in `pyproject.toml` is the same value) |
| `eval-regression` | after `test`: `check_regression.py` on fixture data, then on `data/chunks_large.json` (BM25-only, threshold 0.95) |
| `gitleaks` | secret scan (reusable workflow) |

Required check on `main`: `gitleaks`. The other jobs (`markdownlint`, `test`, `eval-regression`) are expected to pass before merging.

## Commit style

```text
<type>(<scope>): <summary>

Closes #<issue>
```

Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`.
