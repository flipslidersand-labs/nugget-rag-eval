# CLAUDE.md

nugget-rag-eval: 検索結果から「チャンク全体」と「クエリに近い文だけ（nugget）」を返したときの Recall・語数を比べる評価ハーネス。
**public リポ**。構成は [docs/design.md](docs/design.md)、引数とデータ形式は [docs/spec](docs/spec/cli-and-data-formats.md)、規約は [CONTRIBUTING.md](CONTRIBUTING.md)。

## コマンド

```bash
pip install -e ".[dev]"                       # Python 3.12+。eval/ と nugget_rag/ を import できるようになる
pytest                                        # カバレッジ fail_under = 90
ruff check . && ruff format --check .         # line-length = 100、ruff は >=0.17,<0.18 に固定
python eval/check_regression.py --chunks tests/fixtures/chunks_test.json --gold tests/fixtures/gold_test.json --threshold 0.95
```

## 罠

- CLI（`python eval/evaluate.py` など）は **editable install が前提**。入れないと `No module named 'eval'`。`python -I` でも同じ。
- worktree で動かすときは、editable install が本体のチェックアウトを指す。`PYTHONPATH=<worktree の絶対パス>` を付けるか worktree の中で入れ直す。付けないと本体の旧コードを測る。
- 整形は **line-length 100**。自動整形ツールの既定（88）で全体を書き直す PR が出たことがある（#315 を close）。
- `eval/baseline_hard.json` と `eval/baseline_gold.json`（CI の実データゲートが読む）は意図した変更のときだけ更新し、新しい値を PR 本文に書く。
- `data/` は `chunks_large.json` 以外 gitignore（PDF 由来のプールは入れない）。CI の回帰ゲートがこのファイルを読む。
- 論文を足すときは `nugget_rag/paper_registry.py` の `ARXIV_MAP`（arxiv_id → paper_id の唯一の正本）。

## 公開リポとして

- 内部ホスト名・IP・個人メール・ローカルのユーザー名・API キーを、コード・テスト・ドキュメント・Issue に書かない（プレースホルダは `<internal-host>`）。履歴の残存は #297。
- markdownlint が全 `*.md` に掛かる（MD013 は無効）。README は EN+JA を保つ。
- PR 題名は `type(scope): 件名`。`CHANGELOG.md` は題名から作るので、題名で変更が分かるようにする。
