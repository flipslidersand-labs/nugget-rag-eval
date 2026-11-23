"""コミット済み gold ファイル自体のスキーマ・レジストリ整合性テスト (#332).

gold_set_hard.json は apply_gold_review.py 等で書き換えられるが、CI の実データゲートは
gold_set.json しか通らない。arxiv_id の誤記・空 span・重複 query が混入しても気づける
ように、実ファイルを validate_gold / ARXIV_MAP / baseline と突き合わせる。
件数のハードコードはせず、baseline の n との照合に留める。
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from eval.evaluate import validate_gold
from nugget_rag.paper_registry import ARXIV_MAP, resolve_paper_key

REPO_ROOT = pathlib.Path(__file__).parent.parent
EVAL_DIR = REPO_ROOT / "eval"

# gold ファイル名 -> 対応する baseline ファイル名
GOLD_BASELINES = {
    "gold_set.json": "baseline_gold.json",
    "gold_set_hard.json": "baseline_hard.json",
}


def _load(name: str):
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def unregistered_items(gold: list[dict]) -> list[tuple[int, object]]:
    """resolve_paper_key が ARXIV_MAP の値(登録済み paper_id)にならない item の (index, key)。"""
    registered = set(ARXIV_MAP.values())
    return [
        (i, resolve_paper_key(item))
        for i, item in enumerate(gold)
        if resolve_paper_key(item) not in registered
    ]


def duplicate_queries(gold: list[dict]) -> list[str]:
    """前後の空白を除いて重複している query (重複したもののみ、初出順)。"""
    seen: set[str] = set()
    dups: list[str] = []
    for item in gold:
        q = item["query"].strip()
        if q in seen and q not in dups:
            dups.append(q)
        seen.add(q)
    return dups


@pytest.mark.parametrize("name", list(GOLD_BASELINES))
def test_gold_file_passes_validate_gold(name):
    validate_gold(_load(name))


@pytest.mark.parametrize("name", list(GOLD_BASELINES))
def test_gold_file_all_items_resolve_to_registered_paper(name):
    assert unregistered_items(_load(name)) == []


@pytest.mark.parametrize("name", list(GOLD_BASELINES))
def test_gold_file_queries_are_unique(name):
    assert duplicate_queries(_load(name)) == []


@pytest.mark.parametrize("name", list(GOLD_BASELINES))
def test_gold_file_count_matches_baseline_n(name):
    baseline_path = EVAL_DIR / GOLD_BASELINES[name]
    if not baseline_path.is_file():
        if name == "gold_set_hard.json":
            pytest.fail(f"{baseline_path.name} が存在しない")
        pytest.skip(f"{baseline_path.name} 未コミット(baseline 照合は存在する場合のみ)")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["n"] == len(_load(name))


# ── 検査自体が不正を検出できること(合成データ。実ファイルは変更しない) ──────────


def _valid_gold() -> list[dict]:
    arxiv_id = next(iter(ARXIV_MAP))
    return [
        {"query": "q one", "arxiv_id": arxiv_id, "answer_spans": ["span a"]},
        {"query": "q two", "arxiv_id": arxiv_id, "answer_spans": ["span b"]},
    ]


def test_checks_accept_valid_synthetic_gold():
    gold = _valid_gold()
    validate_gold(gold)
    assert unregistered_items(gold) == []
    assert duplicate_queries(gold) == []


def test_checks_detect_mistyped_arxiv_id():
    gold = _valid_gold()
    gold[1]["arxiv_id"] = "2410.10070"  # 誤記 (登録は 2410.10071)
    validate_gold(gold)  # スキーマ上は通るので、レジストリ照合でのみ検出される
    assert unregistered_items(gold) == [(1, "2410.10070")]


def test_checks_detect_empty_span():
    gold = _valid_gold()
    gold[0]["answer_spans"] = []
    with pytest.raises(ValueError, match="answer_spans"):
        validate_gold(gold)


def test_checks_detect_duplicate_query():
    gold = _valid_gold()
    gold[1]["query"] = "  " + gold[0]["query"]
    validate_gold(gold)  # スキーマ上は通る
    assert duplicate_queries(gold) == [gold[0]["query"]]


def test_checks_detect_corruption_of_real_gold_copy():
    """実 gold のコピー(メモリ上)に不正を入れると各検査が落ちる。"""
    gold = copy.deepcopy(_load("gold_set_hard.json"))
    assert len(gold) >= 2
    gold[0]["arxiv_id"] = gold[0]["arxiv_id"][:-1] + "x"
    gold[1]["query"] = gold[0]["query"]
    gold[1]["answer_spans"] = []
    assert unregistered_items(gold) == [(0, gold[0]["arxiv_id"])]
    assert duplicate_queries(gold) == [gold[0]["query"]]
    with pytest.raises(ValueError, match="answer_spans"):
        validate_gold(gold)
