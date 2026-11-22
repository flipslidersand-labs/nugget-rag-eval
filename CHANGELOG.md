# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version comes from git tags.

Entries are generated from merged pull-request titles (`feat` → Added, `refactor` → Changed,
`fix` → Fixed). `docs`, `test`, `chore` and `ci` pull requests are not listed; see
[the merged pull requests](https://github.com/flipslidersand-labs/nugget-rag-eval/pulls?q=is%3Apr+is%3Amerged) for those.

## [Unreleased]

Merged after v2.0.0 (2026-08-21).

### Added

- 難問ゴールドを人の確認に基づき35件に絞り、基準値を再記録 ([#223](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/223))
- nugget の失敗分類と掃引を再現する analyze_nugget_failures.py ([#221](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/221))
- README の表を測定から生成する make_eval_report.py ([#220](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/220))
- 確認シートの判定を gold に適用する apply_gold_review.py ([#219](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/219))
- embed-weight 掃引を、文の埋め込みを1回だけ計算して回すスクリプトに ([#216](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/216))
- 実チャンク全文の取得と正規化照合を追加し、プール比較を README に反映 ([#214](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/214))
- full/nugget の paired 比較(McNemar・bootstrap・効率比)を追加 ([#207](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/207))
- 難問セットの基準値からの低下チェックと latency 計測を追加 ([#206](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/206))
- 難問ゴールドセットと PDF 由来の候補プール作成スクリプトを追加 ([#197](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/197))
- nugget に前後文の窓(--nugget-window)を opt-in で追加 ([#204](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/204))
- PRテンプレートにtest-generator呼び出しチェック項目を追加 ([#172](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/172))
- CIにgitleaksシークレットスキャンを追加 ([#171](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/171))
- **embedder**: 指数バックオフリトライ (Closes #85) ([#97](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/97))
- **embedder**: バッチ分割 MAX_BATCH_SIZE=256 (Closes #84) ([#96](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/96))
- eval パッケージ化 + sys.path.insert 除去 ([#99](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/99))
- **check_regression**: --verbose フラグ追加 (Closes #86) ([#93](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/93))

### Changed

- cosine/embed_scores を vectors.py に移し scorer の embedder 依存を解消 ([#213](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/213))
- evaluate の2モード直書きを MODES 表のループに一般化 ([#210](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/210))
- CLI ヘルパーを eval/cli_helpers.py に集約し重複を解消 ([#209](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/209))
- EmbedClient から死んだ api_key 引数を削除 ([#188](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/188))
- extract ARXIV_MAP into nugget_rag/paper_registry ([#98](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/98))
- **tests**: 共通ヘルパーを tests/conftest.py に集約 (Closes #88) ([#91](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/91))

### Fixed

- **check_regression**: baseline のモード不一致と top_k/gold/n の不一致を FAIL にする ([#312](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/312))
- **embedder**: 制御文字を含む API キーをキー値を出さずに EmbedError で止める ([#311](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/311))
- **fetch_papers**: 5xx 再試行で HTTPError を close し main の部分書き出しをテスト ([#310](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/310))
- **scripts**: 失敗箇所表示・analyze の欠け件数・sweep の例外捕捉 ([#309](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/309))
- **check_regression**: raw_recall の hits 型防御と README への hits 追記 ([#308](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/308))
- **scorer**: _tokenize の結合濁点で bigram の並びを切らない ([#296](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/296))
- 408/429 を再試行し Retry-After を上限付きで反映 ([#300](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/300))
- validate_gold rejects paper_id null via shared has_paper_id ([#299](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/299))
- **chunker**: 略語・図表参照の直後で文を割らない ([#298](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/298))
- **ci**: 孤立した qa-workflows の SHA 固定を同内容の新コミットへ差し替える ([#307](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/307))
- **apply_gold_review**: arxiv_id 欠落の gold を KeyError でなく [ERROR] で終了 ([#294](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/294))
- validate_chunks/validate_gold で text と query の型を検査する ([#290](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/290))
- fetch_papers は 0 件取得時に既存 out を上書きせず exit 1 する ([#289](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/289))
- 丸め前の recall で閾値判定し閾値直下の回帰を FAIL にする ([#288](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/288))
- turn evaluate ValueError into [ERROR] exits in other scripts ([#281](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/281))
- tokenizer Latin/Greek/full-width handling and hyphen/decimal policy ([#275](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/275))
- retry and wrap read() timeouts/disconnects in fetch_papers ([#279](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/279))
- NFKC-normalize text in --match normalized ([#278](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/278))
- ベクトル個数・次元の不一致を ValueError にする ([#274](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/274))
- validate API response value types in fetch_papers ([#273](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/273))
- embedder の不正 UTF-8 / 非 list vectors を EmbedError に変換 ([#272](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/272))
- exit with [ERROR] on corrupt embed cache in load_cache ([#263](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/263))
- handle pdftotext missing/timeout and skip empty-text PDFs ([#261](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/261))
- _tokenize の句読点処理(BM25 の文末語・非 ASCII 句読点) ([#260](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/260))
- embed_weight が 0 のとき埋め込みサービスを呼ばない ([#256](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/256))
- 未登録 arxiv_id を failure_count に数え paper_id へフォールバックする ([#255](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/255))
- 非ループバックの http:// で X-API-Key を送る際に警告する ([#254](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/254))
- main で EmbedError を捕捉してエラー終了する ([#238](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/238))
- API 応答本文の不正を FetchError に変換する ([#237](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/237))
- validate で空 span と型不正を弾く ([#236](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/236))
- embedder の read 中の例外をリトライし EmbedError にする ([#235](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/235))
- 誤って外した難問 11・16 を復元し、11 の span を共通部分に変える ([#226](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/226))
- △判定の4件のクエリ・span を直し、基準値を再記録 ([#225](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/225))
- split_sentences が小数点を割る問題を修正 ([#203](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/203))
- arxiv_id→paper_key解決をchunksとgoldで統一 ([#190](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/190))
- CIをself-hosted runner対応に変更 ([#174](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/174))
- add markdownlint config and fix MD040 violations ([#169](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/169))
- hybrid tokenizer eliminates cross-language BM25 zero scores ([#165](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/165))
- batch embed_fn calls in retrieve_nuggets to O(1) per query ([#166](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/166))
- harden fetch_papers.py main() error handling ([#167](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/167))
- validate_chunks + build_chunks_index — fail-fast on ID/text schema errors ([#168](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/168))
- use corpus DF (not per-doc TF) in BM25 IDF ([#164](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/164))
- specify encoding=utf-8 for read_text/write_text ([#158](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/158))
- warn on gold paper keys missing from chunks, fail-fast when all missing ([#162](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/162))
- judge recall_at_k per-result to kill cross-chunk-boundary false positives ([#159](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/159))
- block redirect following so X-API-Key never leaves the validated host ([#163](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/163))
- close HTTPError response body before 5xx retry ([#156](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/156))
- add exponential-backoff retry to fetch_papers HTTP calls ([#161](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/161))
- add real-data eval regression gate to CI ([#160](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/160))
- validate --threshold/--top-k ranges in check_regression.py ([#157](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/157))
- narrow weak pytest.raises(Exception) assertions to EmbedError ([#154](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/154))
- remove dead runtime dependency rank-bm25 ([#155](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/155))
- wrap urlopen with context manager to prevent socket leaks ([#135](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/135))
- replace sys.exit with FetchError for partial-failure resilience ([#134](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/134))
- remove --embedding-api-key CLI arg; read api key from env only ([#123](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/123))
- resp["vectors"]/"embeddings" キー不一致を解消 — フォールバックと明確なエラーメッセージを追加 ([#121](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/121))
- replace raw stack traces with friendly error messages on file I/O ([#127](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/127))
- importlib.reload を subprocess.run に置き換えてテスト順序依存を排除 ([#133](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/133))
- coverage 計測範囲を nugget_rag/eval に限定し scripts/ を除外 ([#132](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/132))
- SSRF 対策 — base_url / fetch URL のスキームを http/https に限定 ([#131](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/131))
- CI eval-regression 閾値を 0.5 → 0.95 に引き上げ ([#129](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/129))
- qa-workflows reusable workflow を SHA にピン留め ([#128](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/128))
- validate_gold で必須フィールド欠落・answer_spans 空リストを early exit ([#126](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/126))
- _mock_urlopen デッドコードを削除 ([#125](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/125))
- eval-regression に needs:test 追加 + pip キャッシュ有効化 ([#124](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/124))
- add range validation for --embed-weight, --top-k, --large-chunk-target ([#122](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/122))
- _normalize([]) 空リスト ValueError — ガード追加 ([#120](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/120))
- write chunks.json atomically via tmp + replace() ([#119](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/119))
- move key_field out of loop in fetch_per_query ([#118](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/118))
- **fetch_papers**: urlopen に try/except 追加 (Closes #83) ([#94](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/94))
- **security**: --embedding-api-key → EMBEDDING_API_KEY 環境変数優先に変更 (Closes #58) ([#75](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/75))
- **embedder**: EmbedError 追加・urlopen/json エラーをラップ (Closes #59) ([#74](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/74))
- **check_regression**: arxiv_id チャンクの KeyError 修正・evaluate.py と統一 (Closes #61) ([#73](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/73))
- **fetch_papers**: マージコンフリクト解消・arxiv_id フィールド付与 (Closes #56) ([#72](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/72))
- **evaluate**: arxiv_id → ARXIV_MAP → paper_id 解決で Recall = 0 を修正 (#39 #40) ([#43](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/43))

## [2.0.0] - 2026-08-21

Release title: "MRR@k・CJK対応・arxiv_id移行・CI回帰テスト" ([release](https://github.com/flipslidersand-labs/nugget-rag-eval/releases/tag/v2.0.0)).

### Added

- **gold-set**: paper_id を arxiv_id (stable) に移行 ([#27](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/27))
- **eval**: MRR@k メトリクスを追加 ([#24](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/24))
- **ci**: Recall@5 閾値チェックを CI に追加 ([#29](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/29))
- **scorer**: BM25 + e5-base ハイブリッド nugget スコアリング ([#10](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/10))

### Fixed

- **eval**: --token-estimator オプションを追加（words/chars モード） ([#28](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/28))
- **scorer**: 日本語向け文字 n-gram tokenizer を追加 ([#25](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/25))
- **eval**: --embedding-collection 引数を追加 ([#26](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/26))
- **chunker**: CJK 文末記号（。！？）対応 ([#23](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/23))
- **scorer**: top_nuggets を元文脈順に保持 ([#22](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/22))
- **retriever**: BM25クエリランキング実装 + per-query fetch ([#7](https://github.com/flipslidersand-labs/nugget-rag-eval/pull/7))

[Unreleased]: https://github.com/flipslidersand-labs/nugget-rag-eval/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/flipslidersand-labs/nugget-rag-eval/releases/tag/v2.0.0
