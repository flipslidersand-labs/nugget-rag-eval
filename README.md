# nugget-rag-eval

CoinRAG ([arxiv 2608.07458](https://arxiv.org/abs/2608.07458)) のコアアイデア —
**チャンク全体でなくクエリ関連スパン（ナゲット）だけを LLM に渡す** — を
academic-paper-system に導入する前に定量評価するためのリポジトリ。

## 仮説

> 512トークンのチャンク全体よりも、チャンク内の上位1〜2文だけを返す方が
> Recall@5 が高く、かつコンテキスト長が短くなる。

## 評価設計

```
クエリ → hybrid search (full-chunk) → snippets   →  Recall@k 計測
                                                   ↕  比較
クエリ → hybrid search → nugget抽出 → nuggets     →  Recall@k 計測
```

## ディレクトリ構成

```
nugget_rag/
  chunker.py     # 文単位分割（CJK 対応）
  scorer.py      # BM25 + 埋め込み類似度でナゲットスコアリング
  retriever.py   # full-chunk / nugget 両モードのリトリーバー
  embedder.py    # 外部 embedding サービスクライアント
eval/
  gold_set.json  # クエリ + 正解スパン（手作りゴールドセット）
  evaluate.py    # Recall@k / MRR@k 計算
  check_regression.py  # CI 回帰チェック（Recall 閾値監視）
scripts/
  fetch_papers.py  # academic-paper-system API から論文チャンクを取得
tests/
  test_chunker.py
  test_scorer.py
  test_evaluate.py
  test_check_regression.py
```

## セットアップ

```bash
pip install -e ".[dev]"
```

## 実行

```bash
# 1. 論文チャンクを取得（推奨: per-query fetch）
python scripts/fetch_papers.py \
  --api-url http://localhost:8020 \
  --gold-set eval/gold_set.json \
  --chunk-mode large \
  --out data/chunks_large_perquery.json

# または generic query で一括取得（小チャンク）
python scripts/fetch_papers.py --api-url http://localhost:8020 --out data/chunks.json

# 2. 評価実行（BM25 のみ）
python eval/evaluate.py \
  --chunks data/chunks_large_perquery.json \
  --gold eval/gold_set.json \
  --token-estimator words

# 3. BM25 + embedding ハイブリッド
python eval/evaluate.py \
  --chunks data/chunks_large_perquery.json \
  --gold eval/gold_set.json \
  --embedding-url http://<internal-host>:9092 \
  --embedding-api-key <key> \
  --embed-weight 0.5 \
  --token-estimator words
```

## 出力フォーマット

```
Mode               Recall@5     MRR@5        Avg tokens(words)
------------------------------------------------------------
full-chunk         1.0          0.823        300.1
nugget             1.0          0.791        100.5

{
  "n_queries": 19,
  "full_chunk": {"recall": 1.0, "mrr": 0.823, "avg_tokens": 300.1},
  "nugget":     {"recall": 1.0, "mrr": 0.791, "avg_tokens": 100.5}
}
```

## CI 回帰チェック

```bash
python eval/check_regression.py \
  --chunks data/chunks_large.json \
  --gold eval/gold_set.json \
  --threshold 0.95
```

Recall@5 がいずれかのモードで閾値を下回ると exit 1 で CI を落とす。

## 評価指標

| 指標 | 説明 |
| ------ | ------ |
| **Recall@k** | 上位 k 件の中に正解スパンが含まれるかどうかの割合（ヒット/ミスの2値）。順位は考慮しない。 |
| **MRR@k** | Mean Reciprocal Rank。クエリごとに最初にヒットした順位の逆数（1/rank）を平均したもの。1位でヒット→1.0、2位→0.5、3位→0.333…。Recall@k が同じでも、より上位でヒットするほど MRR は高くなる。 |
| **Avg tokens** | 取得結果の平均トークン数。`words`（split）または `chars`（len/4）で推定可能。 |

## 評価結果（実測値）

### 大チャンク + per-query fetch（現行・推奨）

19クエリ・10論文（arxiv）・gold set 実チャンクテキスト検証済み。

```
Mode               Recall@5     MRR@5        Avg tokens(words)
------------------------------------------------------------
full-chunk         1.000        0.823        300.1
nugget             1.000        0.791        100.5        (削減率 66.5% ✅)
```

embedding モデル: intfloat/multilingual-e5-base（MINIPC embedding-svc :9092）

**仮説検証：** BM25 + per-query fetch により nugget が full-chunk と同等の Recall 1.0 を達成。
context length を **66.5%** 削減しながら Recall を完全維持。
academic-paper-system への nugget 導入を推奨。

**embed-weight について:** 現在の `--embed-weight 0.5` は仮置き値。Recall@5=1.0 が出ている段階でチューニングを止め、次フェーズで MRR と latency のトレードオフを見ながら最適化する想定。0.3〜0.7 の範囲で比較予定（Issue #11）。

### 小チャンク（参考・旧方式）

```
Mode               Recall@5     MRR@5        Avg tokens(words)
------------------------------------------------------------
full-chunk         0.211        -            24.1
nugget             0.211        -            23.2         (削減率 3.7%)
```

チャンク自体が小さい（~24 tokens）ため nugget 削減効果が軽微。大チャンクモード推奨。
