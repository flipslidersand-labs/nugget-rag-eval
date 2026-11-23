# Spec: CLIs and data formats / 仕様: CLI とデータ形式

外から見た振る舞いだけを書く。内部の構造は [design.md](../design.md)、使い方の流れは [README](../../README.md)。
各 CLI の最新の引数は `--help` が正本（venv に `pip install -e ".[dev]"` した状態で実行する）。

What the tools accept and produce. Internals: [design.md](../design.md). Usage walk-through: [README](../../README.md).
`--help` is the source of truth for flags (run it after `pip install -e ".[dev]"`).

## データ形式 / Data formats

### チャンク（`--chunks`）

JSON 配列。各要素は次を持つ。

| キー | 必須 | 内容 |
| --- | --- | --- |
| `text` | 必須 | 空白だけでない文字列 |
| `arxiv_id` または `paper_id` | どちらか必須 | 論文の識別子。両方あるとき `arxiv_id` が優先される |
| `chunk_indices`, `score` など | 任意 | 取得スクリプトが付ける。評価は読まない |

`arxiv_id` を内部の `paper_id` に対応づける表は `nugget_rag/paper_registry.py` の `ARXIV_MAP`。
検証に失敗すると `[ERROR] chunks[i] ...` を出して終了コード 1。

### ゴールド（`--gold`）

JSON 配列。各要素は次を持つ。

| キー | 必須 | 内容 |
| --- | --- | --- |
| `query` | 必須 | 空白だけでない文字列 |
| `answer_spans` | 必須 | 空でない文字列の配列。空白だけの要素は不可 |
| `arxiv_id` または `paper_id` | どちらか必須 | 正解論文。この論文のチャンクだけから検索する |
| `notes` | 任意 | 評価は読まない |

ヒットの定義: 上位 k 件の結果のテキストのどれかに、いずれかの `answer_spans` が含まれること。
照合は `--match exact`（大文字小文字を無視した部分文字列、既定）か `--match normalized`（空白・記号も無視）。

同梱: `eval/gold_set.json`（19 件）、`eval/gold_set_hard.json`（難問。件数は README と ADR-003 を参照）。

### 基準値（`check_regression.py --baseline`）

`eval/baseline_hard.json` の形。`full_chunk.recall` と `nugget.recall`（モードごとの記録値）に加えて、
`top_k`・`gold`・`n`・`nugget_window`・`pool`・`_comment` を持つ。
モード名が 1 つも一致しない、または `top_k`・`gold`・`n` が今回の実行と違うときは FAIL（キーが無い古い形式は照合しない）。
`--advisory` のときは FAIL ではなく `[WARN]`（終了コード 0）で、「比較不能」と出す。
ファイルが無いときは低下チェックを飛ばす。

## CLI

### `eval/evaluate.py`

チャンクとゴールドを読み、full-chunk と nugget の Recall@k・MRR@k・平均トークン数を表と JSON で出力する。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `--chunks`, `--gold` | 必須 | 上記の形式。存在しないパスは引数エラー |
| `--top-k` | 5 | 正の整数 |
| `--token-estimator` | `words` | `words`（`split()`）か `chars`（`len/4`） |
| `--nugget-window` | 0 | nugget の前後 N 文も返す。0 は無効 |
| `--match` | `exact` | `exact` か `normalized` |
| `--pool-label` | なし | 候補プールの作り方のメモ。JSON の `pool` にそのまま出る |
| `--embedding-url` | なし | 指定すると nugget の文選びが BM25 + 埋め込みのハイブリッドになる |
| `--embed-weight` | 0.5 | 0 = BM25 のみ、1 = 埋め込みのみ。0 のときは埋め込みサービスを呼ばない |
| `--embedding-collection` | `search-engine` | 埋め込みサービスのコレクション名 |
| `--verbose` | off | 失敗クエリの詳細を出す |

出力の例は [README の出力フォーマット](../../README.md#出力フォーマット)。JSON は次のキーを持つ。

| キー | 内容 |
| --- | --- |
| `match` | 使った照合方式（`exact` か `normalized`） |
| `n_queries` | gold のクエリ数 |
| `n_queries_without_chunks` | gold の論文キーに対応するチャンクが `--chunks` に無いクエリ数。そのクエリは静かに Recall 0 になるので、0 でなければ標準エラーに `[WARN]` も出る。全クエリが該当なしのときは Recall 0 を出さず `[ERROR]` で終了コード 1 |
| `modes` | 評価したモード名の配列（`full_chunk`, `nugget`） |
| `hits` | モードごとの正解クエリ数（丸め前） |
| `full_chunk`, `nugget` | モードごとの `recall`・`mrr`・`avg_tokens` |
| `latency_ms` | モードごとのレイテンシ（`p50`・`p95`） |
| `comparison` | full-chunk と nugget の対応あり比較 |
| `pool` | `--pool-label` を指定したときだけ出る。指定した文字列そのまま |

入力の検証エラーは `[ERROR] ...` を標準エラーに出して終了コード 1。

### `eval/check_regression.py`

Recall が下がっていないかを見る CI 用のゲート。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `--chunks`, `--gold` | 必須 | 同上 |
| `--threshold` | 0.95 | 0〜1。いずれかのモードの Recall@k がこれ未満なら失敗 |
| `--top-k` | 5 | |
| `--baseline` | なし | 記録した Recall の JSON |
| `--max-drop` | 0.05 | 基準値からの許容低下幅 |
| `--advisory` | off | 基準値からの低下を `[WARN]` で出して終了コード 0。基準の `top_k`・`gold`・`n` が今回と違う、またはモード名が 1 つも一致しない場合も `[WARN]`（終了コード 0）で、「比較不能」と併記する |
| `--verbose`, `-v` | off | 失敗クエリの詳細を標準エラーに出す |

終了コード: 閾値未満 → 1（`[FAIL]`）。基準値からの低下、または基準の `top_k`・`gold`・`n` の不一致、またはモード名が 1 つも一致しない → 1（`--advisory` なら 0。不一致は比較不能なので数値の差は意味を持たない）。入力不正 → 1（`[ERROR]`）。通れば `[PASS]` で 0。
閾値の判定は丸め前の Recall で行う。

### `scripts/fetch_papers.py`

論文検索 API からチャンクを集めて JSON に書く。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `--api-url` | `http://localhost:8020` | 検索 API |
| `--out` | `data/chunks.json` | 出力先。0 件のときは既存ファイルを上書きせず終了コード 1 |
| `--gold-set` | なし | 指定するとゴールドのクエリごとに取得する（正解論文のチャンクが候補に入る） |
| `--query` | `method results contribution` | `--gold-set` が無いときの共通クエリ |
| `--chunk-mode` | `small` | `small`（約 20 トークン）か `large`（約 512 トークンに結合） |
| `--large-chunk-target` | 512 | `large` のときの目標トークン数 |
| `--snippet-length` | 200 | API が返すテキストの最大文字数。**0 で全文**。既定の 200 では各チャンクの先頭しか来ない |
| `--fail-fast` | off | 最初の取得エラーで止める |
| `--max-failures` | 0 | 失敗数（取得エラー + 未登録の `arxiv_id`）がこれを超えると終了コード 1 |

### `scripts/build_pool_from_pdfs.py`

ローカルの PDF から、クエリに依存しない候補プールを作る。外部コマンド `pdftotext`（poppler-utils）が要る。

| 引数 | 既定 | 内容 |
| --- | --- | --- |
| `--pdf-dir` | `data/papers` | PDF の置き場所。ファイル名は `<arxiv_id>.pdf`。`ARXIV_MAP` に無いものは警告して飛ばす |
| `--out` | `data/chunks_pdf.json` | 出力先 |
| `--target-words` | 300 | 1 チャンクあたりの目標語数 |

### その他のスクリプト

| スクリプト | 役割 |
| --- | --- |
| `scripts/analyze_nugget_failures.py` | nugget が外した理由を A 検索ミス / B 分割ミス / C 抽出ミス / D ヒットに分類し、`--sweep` で `nuggets_per_chunk` × 窓を掃引する |
| `scripts/sweep_embed_weight.py` | `--embed-weight` を振って再評価する。文の埋め込みは 1 回だけ計算してキャッシュする。`--embedding-url` と `--out-dir` が必須 |
| `scripts/make_eval_report.py` | README の評価表を測定から生成する。`--pool NAME=PATH[:exact\|normalized]` と `--gold NAME=PATH` を複数指定する |
| `scripts/apply_gold_review.py` | 人の確認シートの判定（○△×）を難問ゴールドに適用する。既定は dry run で、`--write` か `--out` で書く |

## 環境変数 / 外部サービス

| 名前 | 使う場所 | 内容 |
| --- | --- | --- |
| `EMBEDDING_API_KEY` | `evaluate.py`・`sweep_embed_weight.py` | 埋め込みサービスの鍵。`X-API-Key` ヘッダで送る。CLI 引数では受け取らない |

埋め込みサービス（`--embedding-url`）は `POST /embed/batch` に `{"texts": [...], "collection": "<name>"}` を受け、
`{"vectors": [[...]]}`（または `{"embeddings": [[...]]}`）を返す想定。リモートへの `http://` は鍵が平文で流れるため警告が出る。

クライアント（`nugget_rag/embedder.py`）の制約:

- URL は `http`・`https` のみで、ホストが必須。`user:pass@host` 形式は拒否する（鍵は `EMBEDDING_API_KEY` で渡す）。リダイレクトは辿らない。
- `texts` は 256 件ずつに分けて送る（`MAX_BATCH_SIZE`）。
- 応答本文が 32 MiB を超えたら、パースせずに `EmbedError` で止める（`MAX_RESPONSE_BYTES`）。
