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
# 1. 論文チャンクを取得
python scripts/fetch_papers.py --api-url http://localhost:8020 --out data/chunks.json

# 2. ゴールドセットを編集（eval/gold_set.json）

# 3. 評価実行
python eval/evaluate.py --chunks data/chunks.json --gold eval/gold_set.json
```

## 出力例

```
Mode           Recall     Avg tokens
------------------------------------
full-chunk     0.400      512.0
nugget         0.550      64.0
```
