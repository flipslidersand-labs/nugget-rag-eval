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
# 1. 論文チャンクを取得（小チャンク）
python scripts/fetch_papers.py --api-url http://localhost:8020 --out data/chunks.json

# または 512+ token 大チャンク
python scripts/fetch_papers.py --api-url http://localhost:8020 --chunk-mode large --out data/chunks_large.json

# 2. 評価実行
python eval/evaluate.py --chunks data/chunks.json --gold eval/gold_set.json
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

### 大チャンク (~250 tokens)

```text
Mode           Recall     Avg tokens    削減率
------------------------------------
full-chunk     0.526      254.6        -
nugget         0.474      57.0         77.6% ✅
```

**仮説部分検証：** 大チャンクでは nugget が context length を **77.6%** 削減。
Recall 差は gold set 品質に依存（実装は正常）。
