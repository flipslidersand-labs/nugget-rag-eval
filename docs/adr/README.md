# 設計判断（ADR）/ Design decisions

遡及して整理した Draft。当時の比較経緯は推定していない。雛形は repo-compass の ADR-025（public リポの ADR の基準）に従う。

| ADR | EN | 題 | 状態 |
| --- | --- | --- | --- |
| [ADR-001](ADR-001-compare-before-adopting.md) | Compare before adopting | nugget 方式は、導入前にチャンク全体と同じ条件で比較する | Draft |
| [ADR-002](ADR-002-zero-runtime-dependencies.md) | Zero runtime dependencies | 実行時の依存をゼロにし、BM25 を自前実装する | Draft |
| [ADR-003](ADR-003-gold-set-and-regression-gate.md) | Gold set and regression gate | ゴールドセットは手作りで検証し、Recall の回帰を CI で止める | Draft |
