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
  chunker.py     # 文単位分割
  scorer.py      # BM25 + 埋め込み類似度でナゲットスコアリング
  retriever.py   # full-chunk / nugget 両モードのリトリーバー
eval/
  gold_set.json  # クエリ + 正解スパン（手作りゴールドセット）
  evaluate.py    # Recall@k / MRR 計算
scripts/
  fetch_papers.py  # academic-paper-system API から論文チャンクを取得
tests/
  test_chunker.py
  test_scorer.py
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

# 2. 評価実行
python eval/evaluate.py --chunks data/chunks_large_perquery.json --gold eval/gold_set.json
```

## 評価結果（実測値）

### 小チャンク (~20 tokens)

```text
Mode           Recall     Avg tokens
------------------------------------
full-chunk     0.421      21.9
nugget         0.421      19.9
```

削減効果わずか（8.7%）。チャンク自体が小さいため。

### 大チャンク + generic fetch（旧方式）

```text
Mode           Recall     Avg tokens    削減率
------------------------------------
full-chunk     0.526      254.6        -
nugget         0.474      57.0         77.6%
```

retriever がクエリを無視して先頭 k チャンクを返す実装バグにより nugget Recall が低く出ていた。

### 大チャンク + per-query fetch（現行・推奨）

```text
Mode                Recall     Avg tokens    削減率
-----------------------------------------------
full-chunk          0.526      248.9        -
nugget (BM25-only)  0.526      79.9         67.9%
nugget (e5 w=0.3)   0.526      66.1         73.4%
nugget (e5 w=0.5)   0.526      65.2         73.8%
nugget (e5 w=0.7)   0.526      61.8         75.2% ✅ best
```

embedding モデル: intfloat/multilingual-e5-base（MINIPC embedding-svc :9092）

**仮説検証：** BM25 クエリランキング修正 + per-query fetch により nugget Recall が
full-chunk と同率（0.526）に改善。BM25 + e5-base ハイブリッドスコアリング（w=0.7）で
context length を **75.2%** 削減しながら Recall を完全維持。

**推奨設定**: `--embed-weight 0.7` — Recall を維持しながら最大トークン削減。
