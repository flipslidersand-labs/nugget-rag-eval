# nugget-rag-eval

CoinRAG ([arxiv 2608.07458](https://arxiv.org/abs/2608.07458)) のコアアイデア —
**チャンク全体でなくクエリ関連スパン（ナゲット）だけを LLM に渡す** — を
academic-paper-system に導入する前に定量評価するためのリポジトリ。

## 設計判断（ADR）/ Design decisions

- [Compare before adopting](docs/adr/ADR-001-compare-before-adopting.md) — 導入前に同じ条件で比較する
- [Zero runtime dependencies](docs/adr/ADR-002-zero-runtime-dependencies.md) — 実行時の依存ゼロ、BM25 は自前実装
- [Gold set and regression gate](docs/adr/ADR-003-gold-set-and-regression-gate.md) — 手作りのゴールドセットと Recall の回帰チェック

## 仮説

> 512トークンのチャンク全体よりも、チャンク内の上位1〜2文だけを返す方が
> Recall@5 が高く、かつコンテキスト長が短くなる。

**検証状況（2026-10-28 時点）:** 前半（Recall が高い）は**成り立たない**。nugget は上位チャンクの中から文を選ぶため、
nugget が当たる問題は full-chunk も必ず当たる（部分集合）。Recall で full-chunk を上回ることはない。
後半（コンテキストが短い）は成り立つ（語数は約 28%）。つまり問いは「語数を減らす代わりに Recall をどれだけ失うか」で、
難問ではその低下が大きい（[評価結果](#評価結果実測値)・[限界](#限界)）。

## 評価設計

```text
クエリ → hybrid search (full-chunk) → snippets   →  Recall@k 計測
                                                   ↕  比較
クエリ → hybrid search → nugget抽出 → nuggets     →  Recall@k 計測
```

## ディレクトリ構成

```text
nugget_rag/
  chunker.py        # 文単位分割（CJK 対応）
  scorer.py         # BM25 + 埋め込み類似度でナゲットスコアリング
  retriever.py      # full-chunk / nugget 両モードのリトリーバー
  embedder.py       # 外部 embedding サービスクライアント
  paper_registry.py # arxiv_id ↔ paper_id マッピング（ARXIV_MAP）
eval/
  gold_set.json       # クエリ + 正解スパン（既存 19 件。CI 回帰ゲート用）
  gold_set_hard.json  # 難問 37 件（人の確認後。10 論文）。語彙を共有しない言い換えクエリ
  baseline_hard.json  # 難問セットの Recall 基準値（低下幅チェック用）
  evaluate.py         # Recall@k / MRR@k / latency / full-nugget の paired 比較
  compare.py          # McNemar 正確確率・ブートストラップ区間・効率比
  check_regression.py # 回帰チェック（Recall 閾値 / 基準値からの低下幅）
scripts/
  fetch_papers.py          # academic-paper-system API から論文チャンクを取得
  build_pool_from_pdfs.py  # ローカル PDF から query 非依存の候補プールを作る
tests/
  test_chunker.py
  test_compare.py
  test_nugget_window.py
  test_hard_regression_and_latency.py
  test_build_pool_from_pdfs.py
  test_scorer.py
  test_evaluate.py
  test_evaluate_main.py
  test_check_regression.py
  test_embedder.py
  test_fetch_papers.py
  test_no_runtime_deps.py
  test_paper_registry.py
  test_retriever.py
  test_token_estimator.py
  fixtures/  # 評価スクリプト用テストデータ
```

## セットアップ

runtime 依存ゼロ（BM25 は `nugget_rag/scorer.py` に自前実装）。開発用ツールのみ optional dependency として導入する。

```bash
pip install -e ".[dev]"
```

## 実行

```bash
# 1. 論文チャンクを取得（推奨: per-query fetch・チャンク全文）
#    --snippet-length 0 を付けないと API は各チャンクの先頭 200 文字しか返さない
python scripts/fetch_papers.py \
  --api-url http://localhost:8020 \
  --gold-set eval/gold_set.json \
  --chunk-mode large \
  --snippet-length 0 \
  --out data/chunks_large_perquery.json

# または generic query で一括取得（小チャンク）
python scripts/fetch_papers.py --api-url http://localhost:8020 --out data/chunks.json

# 2. 評価実行（BM25 のみ）
python eval/evaluate.py \
  --chunks data/chunks_large_perquery.json \
  --gold eval/gold_set.json \
  --token-estimator words

# 3. BM25 + embedding ハイブリッド
#    キーは環境変数で渡す（CLI 引数に置くと ps/ログに残る）
EMBEDDING_API_KEY=<key> python eval/evaluate.py \
  --chunks data/chunks_large_perquery.json \
  --gold eval/gold_set.json \
  --embedding-url http://<internal-host>:9092 \
  --embed-weight 0.5 \
  --token-estimator words

# 4. 前後の文も返す窓（既定 0 = 無効）
python eval/evaluate.py --chunks <chunks> --gold <gold> --nugget-window 1

# 5. API に依存せず PDF から候補プールを作って難問セットを評価
python scripts/build_pool_from_pdfs.py --pdf-dir data/papers --out data/chunks_pdf.json
python eval/evaluate.py --chunks data/chunks_pdf.json --gold eval/gold_set_hard.json

# 6. API のチャンクは PDF と空白の入り方が違う（"UniversityofCalifornia"）。
#    gold の span を別の抽出器のテキストに当てるときは --match normalized
python eval/evaluate.py --chunks data/chunks_large_perquery.json --gold eval/gold_set_hard.json \
  --match normalized --pool-label "api hybrid per-query, snippet_length=0, 2026-10-28"
```

候補プールの作り方で結果が変わる（[限界](#限界)）。per-query fetch は、gold のクエリで検索した上位だけが候補になる。
`build_pool_from_pdfs.py` は全文を約 300 語ごとに分割するので、クエリに依存しない。

## 出力フォーマット

```text
Mode               Recall@5     MRR@5        Avg tokens(words)
------------------------------------------------------------
full-chunk         0.514        0.373        281.4
nugget             0.216        0.128        78.7

{
  "n_queries": 37,
  "full_chunk": {"recall": 0.514, "mrr": 0.373, "avg_tokens": 281.4},
  "nugget":     {"recall": 0.216,   "mrr": 0.128, "avg_tokens": 78.7},
  "latency_ms": {"full_chunk": {"p50": ..., "p95": ...}, "nugget": {"p50": ..., "p95": ...}},
  "comparison": {
    "counts": {"both": 8, "full_only": 11, "nugget_only": 0, "neither": 18},
    "recall_diff": 0.297, "recall_diff_ci95": [0.162, 0.459], "mcnemar_p": 0.001,
    "token_ratio": 0.28, "recall_retained": 0.421, "distinguishable": true
  }
}
```

（例は難問セット・PDF プール・BM25 のみ。`comparison` の判別条件は `mcnemar_p < 0.05` かつ
`recall_diff_ci95` が 0 を含まないこと。`nugget_only` は構造上常に 0。）

## CI 回帰チェック

```bash
python eval/check_regression.py \
  --chunks data/chunks_large.json \
  --gold eval/gold_set.json \
  --threshold 0.95
```

Recall@5 がいずれかのモードで閾値を下回ると exit 1 で CI を落とす。

### 難問セットの回帰チェック（ローカル実行・助言）

難問（`eval/gold_set_hard.json`）は Recall が約 0.5 のため、固定閾値ではなく**記録した基準値からの低下幅**で見る。
候補プールは論文 PDF の本文を含むためリポに入れない（`data/` は gitignore）。PDF を `data/papers/` に置いて次で作る。

```bash
python scripts/build_pool_from_pdfs.py --pdf-dir data/papers --out data/chunks_pdf.json

python eval/check_regression.py \
  --chunks data/chunks_pdf.json \
  --gold eval/gold_set_hard.json \
  --threshold 0 \
  --baseline eval/baseline_hard.json --max-drop 0.05 --advisory
```

- 基準値は `eval/baseline_hard.json`。意図した変更で動かすときだけ更新し、新しい値を PR 本文に書く。
- `--advisory` は低下を `[WARN]` で出して exit 0（ADR-006: まず助言）。安定したら外して失敗にする。
- 基準ファイルが無いと低下チェックはスキップされる。

`evaluate.py` の出力には `latency_ms`（クエリあたりの p50/p95）が含まれる。embedding を使う場合は `nugget_embed_calls` に埋め込み呼び出しの時間を分けて出す。

## 評価指標

| 指標 | 説明 |
| ------ | ------ |
| **Recall@k** | 上位 k 件の中に正解スパンが含まれるかどうかの割合（ヒット/ミスの2値）。順位は考慮しない。 |
| **MRR@k** | Mean Reciprocal Rank。クエリごとに最初にヒットした順位の逆数（1/rank）を平均したもの。1位でヒット→1.0、2位→0.5、3位→0.333…。Recall@k が同じでも、より上位でヒットするほど MRR は高くなる。 |
| **Avg tokens** | 取得結果の平均トークン数。`words`（split）または `chars`（len/4）で推定可能。 |

## 評価結果（実測値）

### 評価の前提（推奨構成）と、候補プールによる違い（2026-10-28）

実運用に最も近いのは **C: API の per-query hybrid 検索・`--snippet-length 0`（実チャンク全文）・`--match normalized`**。
本番のチャンク分割と検索の結果をそのまま使うため。PDF 全文プール（B）はクエリに依存しない再現用の対照。

同じゴールド・同じ BM25 評価での比較（正解論文のチャンクだけから検索）。難問は人の確認後の 37 件（2026-10-31 に再集計。経緯は[限界](#限界)）:

| プール | チャンク数 | 既存 19: full / nugget Recall | 難問 37: full / nugget Recall | 難問の根拠が候補にある割合 |
|---|---|---|---|---|
| A 旧: per-query hybrid・先頭 200 字の断片 | 63 | 1.000 / 1.000 | 0.027 / 0.027 | 0.027（1/37） |
| **C per-query hybrid・チャンク全文** | 187 | 1.000 / 0.842 | 0.459 / 0.000 | 0.946（正規化照合 35/37） |
| B PDF 全文プール | 354 | 0.842 / 0.789 | 0.514 / 0.216 | 1.000 |

（C は `--match normalized`、A・B は完全一致。A の難問が 0.027 なのは、根拠が先頭 200 字に入らないため。C の完全一致では
根拠の一致が 14/37 にとどまり、英数字で正規化すると 35/37。差は空白の入り方で、内容の欠落ではない。）

- 検索範囲を「正解論文のみ」から「全論文」にすると、さらに下がる（B・難問: full 0.514 → 0.270、nugget 0.216 → 0.108）。
- keyword 検索で候補を作る案は成立しない（複数語クエリで 0 件。40 件版の測定で 59 件中 39 件がチャンクなし）。

実チャンク（C）での full-chunk と nugget の差（`comparison`）:

| ゴールド | 窓 | full / nugget Recall | 保持率 | 語数/full | 判別 |
|---|---|---|---|---|---|
| 既存 19 件 | 0 | 1.000 / 0.842 | 0.84 | 0.175 | できない（p=0.25） |
| 既存 19 件 | 1 | 1.000 / 0.895 | 0.90 | 0.334 | できない（p=0.5） |
| 難問 37 件 | 0 | 0.459 / 0.000 | 0.00 | 0.185 | できる（p<0.0001） |
| 難問 37 件 | 1 | 0.459 / 0.108 | 0.23 | 0.358 | できる（p=0.0002） |

難問では、実チャンクの nugget は PDF プール（B）よりさらに Recall を落とす（0.000 vs 0.216）。保持率は 0%（窓 1 で 24%）。
既存 19 件では逆に C のほうが高い（0.842 vs 0.789）ので、プールの違いの方向は一定ではない。

**確認で件を外した副作用:** 外した 3 件（人の判定）には、C・窓 0 で nugget が当てていた 2 件が含まれていた
（40 件のとき 0.050、37 件で 0.000）。同じ事実が複数の箇所にある件は、どのチャンクを拾っても当たりやすい。外すと nugget の成績は下がる。

### 参考: PDF 全文プール（354 チャンク）・BM25 のみ（2026-10-31 に再集計）

候補プールは `build_pool_from_pdfs.py`（10 論文、約 300 語/チャンク）。評価は正解論文のチャンクだけから検索する。
再現: [実行](#実行)の手順 5。数値は `evaluate.py` の出力による。

| ゴールド | 件数 | モード | Recall@5 | MRR@5 | 平均語数 | 語数/full |
|---|---|---|---|---|---|---|
| 既存 `gold_set.json` | 19 | full-chunk | 0.842 | 0.655 | 278.9 | - |
| | | nugget | 0.789 | 0.532 | 76.7 | 0.275 |
| | | nugget（窓 1） | 0.842 | 0.653 | 146.1 | 0.524 |
| 難問 `gold_set_hard.json` | 37 | full-chunk | 0.514 | 0.373 | 281.4 | - |
| | | nugget | 0.216 | 0.128 | 78.7 | 0.280 |
| | | nugget（窓 1） | 0.378 | 0.259 | 155.6 | 0.553 |

full-chunk と nugget は差があるか（`comparison`。判別条件は実測前に固定: p < 0.05 かつ Recall 差の 95% 区間が 0 を含まない）:

| ゴールド | 窓 | Recall 差（full − nugget）と 95% 区間 | McNemar p | 判別 |
|---|---|---|---|---|
| 既存 19 件 | 0 | 0.053 [0.000, 0.158] | 1.0 | できない |
| 難問 37 件 | 0 | 0.297 [0.162, 0.459] | 0.001 | できる |
| 難問 37 件 | 1 | 0.135 [0.027, 0.243] | 0.0625 | できない |

読み取り:

- 既存 19 件では、nugget は full-chunk と統計的に区別できない（語数は約 28%）。ただし n=19 は検出力が低い。
- 難問では、窓 0 の nugget は Recall を大きく落とす（保持率 0.42）。誤差ではない。窓 1 では保持率 0.74 になり、差は有意でなくなる（p=0.0625。n=37 では検出力が低く、差がないとは言えない）。
- 難問の失敗の内訳（nugget、窓 0）: 検索で根拠チャンクを取れない 18 件、チャンクは取れたが根拠文が上位 3 文に入らない 9 件、
  文分割で根拠が割れた 2 件、成功 8 件（#201。`scripts/analyze_nugget_failures.py` で再現）。検索の失敗は full-chunk も同じなので、nugget だけ直しても Recall 0.514 を超えない。
- 窓 1 は `nuggets_per_chunk` を増やすより効率がよい（難問で 0.216 → 0.378、語数は full の約 55%）。既定は窓 0 のまま。採否は利用側の要件で決まる。

### 旧結果（参考・条件が異なる）: per-query fetch・先頭 200 文字の断片

19 クエリ・10 論文。候補プールは gold のクエリで hybrid 検索した上位のみで、全体で **22 チャンク**。
**この「チャンク」は実チャンクではない。** 検索 API は既定で各チャンクの**先頭 200 文字**だけを返す（`snippet_length=200`）。
旧 `fetch_papers.py` はこれを指定せず、約 14 個の先頭断片を結合して「large チャンク」（約 320 語）にしていた。
512 トークンのチャンク全体に対する結果ではない。

```text
Mode               Recall@5     MRR@5        Avg tokens(words)
------------------------------------------------------------
full-chunk         1.000        0.823        300.1
nugget             1.000        0.791        100.5        (削減率 66.5%)
```

この Recall 1.0 は**断片に対する飽和**で、nugget の有効性の証拠にはならない。gold の span がチャンクの先頭 200 文字から選ばれていたため、
断片だけで当たっていた。実チャンク全文では、同じ 19 件でも nugget は Recall 0.84（窓 0）に下がる（下表）。
以前の「Recall を完全維持して語数を 66.5% 削減、導入を推奨」という結論は言えない。

**embed-weight について:** 旧結果が Recall 1.0 に飽和していたため、`--embed-weight 0.5` は仮置きのままだった（Issue #11）。
実チャンク（プール C）・正規化照合・窓 0 で、重みを振って再評価した（#199。2026-10-28 に確認前の 40 件で測り、2026-10-31 に確認後の 35 件で測り直した。埋め込みはキャッシュを再利用）。
`scripts/sweep_embed_weight.py` は文の埋め込みを 1 回だけ計算して全重みを回す（再現手順は下記）。

| 重み | 既存 19 件 nugget Recall | 難問 35 件 nugget Recall | 難問の nugget 平均語数 |
|---|---|---|---|
| 0.0（BM25 のみ） | 0.842 | 0.000 | 71.3 |
| 0.3 | 0.737 | 0.000 | 68.9 |
| 0.5（旧・仮置き） | 0.737 | 0.000 | 69.4 |
| 0.7 | 0.737 | 0.086 | 59.2 |
| 1.0（埋め込みのみ） | 0.789 | 0.143 | 40.0 |

（この表は 35 件・△修正前の版。37 件での再測定には埋め込み API の鍵が要るため未実施。`scripts/sweep_embed_weight.py` で、キャッシュを使って数分で再測定できる。）

- 実測前から Issue に書いていた例の基準（難問で BM25 のみより +0.1 以上、かつ既存 19 件を下げない）を**両方満たす重みはない**。
  難問は w=1.0 で +0.143 と前半を満たすが、既存 19 件が 0.842 → 0.789 に下がり後半を満たさない。w=0.7 は +0.086 で前半も満たさない。
  仮置きの 0.5 は、難問では BM25 のみと同じ（どちらも 0 件）で、既存 19 件では下がる。
- BM25 のみでは難問の nugget が 1 件も当たらない（0/35）。埋め込みのみ（1.0）で 5 件当たる。ただし 5 クエリぶんの差で検出力が低く、既存 19 件は 1〜2 クエリぶん下がる。有意な改善とも悪化とも言えない。現時点で、埋め込みの併用を採る根拠はない。
- 重みを上げると nugget の語数は減る（0.0 → 1.0 で難問は 71.3 → 40.0）。埋め込みのみ（1.0）は、語数を約 4 割減らしつつ難問の Recall は下げない。
  この語数削減と既存 19 件の低下をどう評価するかは未決。
- 窓 1 や `nuggets_per_chunk` との組み合わせは未測定。埋め込みの呼び出しはクエリあたり約 16〜31 秒かかった（CPU 動作のサービス）。

```bash
EMBEDDING_API_KEY=<key> python scripts/sweep_embed_weight.py --chunks <pool> \
  --gold eval/gold_set.json eval/gold_set_hard.json --embedding-url http://<host>:9092 --out-dir <dir>
```

## 限界

- **正解論文が既知の前提。** 評価は正解論文のチャンクだけから検索する。実運用（全論文から検索）より易しい。全論文プールでの比較は #200。
- **候補プールで結果が変わる。** 同じ 19 件でも、先頭 200 字の断片では nugget Recall 1.0、実チャンクでは 0.842、PDF 全文プールでは 0.789。上の表のとおり、実運用に近い C を前提にする。
- **nugget は構造上 full-chunk を上回れない。** 上位チャンクから文を選ぶ部分集合のため、Recall での優劣は常に「同じ」か「full が上」。
  方式の良し悪しは Recall だけでなく、語数削減との組（`token_ratio` / `recall_retained`）で判断する。採否の閾値は利用側の要件で決まる。
- **ゴールドの規模と作り方。** 既存 19 件＋難問 37 件、10 論文。難問はサブエージェントが 40 件を起草し、span は PDF 由来のプール本文での実在を機械検証した。
  人の確認は途中（#194）。**12 件は人が判定**（○9・×3）、**15 件は Claude が原文を読んで案を出し**（×2・△4・○9。人は未確認）、**13 件は Claude が原文を読んで前提・曖昧さを確認し○のまま（人は未確認）**。
  外したのは人が判定した 3 件（20・22・38）。Claude の案で一度外した 11・16 は復元した（16 は本文と付録 A.1 が同一文で、span が両方に一致するので一意性の問題はなかった=外したのは誤り。11 は文が違うため、span を両方の節に共通する語に変えて残した）。
  △の 4 件（5・14・23・36）は 2026-10-31 に直した（クエリの修正と別 span の追加）。
  **表の数値は、現在の 37 件の再測定値**（2026-10-31）。例外は埋め込み重みの表で、35 件・△修正前の版。人の確認が進めば、件数と数値は変わる。
  AI 単独の査読（独立した Opus）は、人が不採用にした 3 件のうち、×にできたのが 1 件、△が 1 件、見逃しが 1 件で、人が○にした 9 件のうち 5 件に誤警報を出した（表・図・要旨の再掲を「重複」とするため）。
- **span の含有で判定している。** 回答として正しいか、LLM が実際に答えられるかは評価していない（Recall のみ）。
- **本文の表記が API 由来のチャンクと異なる。** 難問の span は `pdftotext` の出力で作った。API のチャンクは空白の入り方が違う（`UniversityofCalifornia`）ので、
  API 由来のチャンクで測るときは `--match normalized` が必要。正規化は空白・記号を無視するだけだが、わずかに偽の一致を増やしうる。
- **小標本。** 既存 19 件では差を検出する力が低い。難問 37 件でも、論文ごとの偏りは未検証。窓 1 の差が有意でなくなったのも、標本が小さいためかもしれない。
- **embedding 併用の測定は 1 条件のみ**（窓 0・`nuggets_per_chunk`=3・正解論文のみ）。上の `latency_ms` の既定表示は BM25 のみの値で、embedding 呼び出しの時間は `nugget_embed_calls` に分かれる。

### 小チャンク（参考・旧方式）

```text
Mode               Recall@5     MRR@5        Avg tokens(words)
------------------------------------------------------------
full-chunk         0.211        -            24.1
nugget             0.211        -            23.2         (削減率 3.7%)
```

チャンク自体が小さい（~24 tokens）ため nugget 削減効果が軽微。大チャンクモード推奨。
