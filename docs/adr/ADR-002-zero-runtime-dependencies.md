# ADR-002: 実行時の依存をゼロにし、BM25 を自前実装する

**ステータス:** Draft  
**決定日:** 未確認（遡及記録: 2026-10-28）  
**review_by:** 2026-11-17

> 雛形: [ADR-000-public-template](https://github.com/flipslidersand-labs/repo-compass/blob/master/docs/adr/ADR-000-public-template.md)。遡及して書いた記録で、当時の理由は推定していない。

---

## 課題

評価用のライブラリに、どの外部パッケージを持ち込むかを決める必要があった。BM25 によるスコアリングが核にある。

## 決定

実行時の依存パッケージをゼロにする。BM25 は `nugget_rag/scorer.py` に自前で実装する。
開発用のツール（pytest・ruff など）だけを optional dependency として導入する。

## 理由

- 実行時の依存がないことを、テスト（`tests/test_no_runtime_deps.py`、#138）で機械的に保証している。
- **却下した代替案:** 外部の BM25 ライブラリを使う。比較した記録はない（当時の理由は未確認）。

## 影響

- BM25 の実装と、その正しさのテストを自分で持つことになる。
- 埋め込みは、ライブラリではなく外部の埋め込みサービス（HTTP）を呼ぶ形で使う。

## 再検討条件

BM25 の実装の保守が重くなったとき、または実行時の依存を許す理由が出たとき。

## 関連

- README「セットアップ」
- `tests/test_no_runtime_deps.py`
