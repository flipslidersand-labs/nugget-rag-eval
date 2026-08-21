# nugget-rag-eval

[![CI](https://github.com/flipslidersand-labs/nugget-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/flipslidersand-labs/nugget-rag-eval/actions/workflows/ci.yml)

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

### 大チャンク + per-query fetch（現行・推奨）

19クエリ・10論文（arxiv）・gold set 実チャンクテキスト検証済み。

```text
Mode           Recall     Avg tokens    削減率
-----------------------------------------------
full-chunk     1.000      300.1        -
nugget         1.000      100.5        66.5% ✅
```

embedding モデル: intfloat/multilingual-e5-base（MINIPC embedding-svc :9092）

**仮説検証：** BM25 + per-query fetch により nugget が full-chunk と同等の Recall 1.0 を達成。
context length を **66.5%** 削減しながら Recall を完全維持。
academic-paper-system への nugget 導入を推奨。

### 小チャンク（参考・旧方式）

```text
Mode           Recall     Avg tokens    削減率
-----------------------------------------------
full-chunk     0.211      24.1         -
nugget         0.211      23.2         3.7%
```

チャンク自体が小さい（~24 tokens）ため nugget 削減効果が軽微。大チャンクモード推奨。
