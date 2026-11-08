import pytest

from nugget_rag.paper_registry import ARXIV_MAP, PAPER_ID_TO_ARXIV, resolve_paper_key


def test_arxiv_map_is_nonempty():
    assert len(ARXIV_MAP) > 0


def test_paper_id_to_arxiv_is_inverse():
    for arxiv_id, paper_id in ARXIV_MAP.items():
        assert PAPER_ID_TO_ARXIV[paper_id] == arxiv_id


def test_paper_ids_are_unique():
    paper_ids = list(ARXIV_MAP.values())
    assert len(paper_ids) == len(set(paper_ids))


def test_arxiv_ids_are_unique():
    arxiv_ids = list(ARXIV_MAP.keys())
    assert len(arxiv_ids) == len(set(arxiv_ids))


def test_known_entry_present():
    assert ARXIV_MAP.get("2608.07458") == 10
    assert PAPER_ID_TO_ARXIV.get(10) == "2608.07458"


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"arxiv_id": "2608.07458"}, 10),
        ({"arxiv_id": "2410.10071"}, 1),
        ({"arxiv_id": "9999.99999"}, "9999.99999"),
        ({"arxiv_id": "9999.99999", "paper_id": 7}, "9999.99999"),
        ({"arxiv_id": "", "paper_id": 7}, 7),
        ({"arxiv_id": None, "paper_id": 7}, 7),
        ({"paper_id": 7}, 7),
        ({"arxiv_id": "2608.07458", "paper_id": 3}, 10),
        ({"arxiv_id": "", "paper_id": None}, None),
        ({"arxiv_id": None}, None),
        ({}, None),
        ({"paper_id": 0}, 0),
        ({"arxiv_id": "", "paper_id": 0}, 0),
        ({"arxiv_id": "2608.07458", "paper_id": 0}, 10),
    ],
    ids=[
        "registered",
        "registered_first",
        "unregistered_returns_string",
        "unregistered_wins_over_paper_id",
        "empty_arxiv_falls_back",
        "none_arxiv_falls_back",
        "missing_arxiv_falls_back",
        "arxiv_wins_over_conflicting_paper_id",
        "empty_arxiv_and_none_paper_id",
        "none_arxiv_no_paper_id",
        "empty_item",
        "paper_id_zero_only",
        "empty_arxiv_paper_id_zero",
        "arxiv_wins_over_paper_id_zero",
    ],
)
def test_resolve_paper_key(item, expected):
    result = resolve_paper_key(item)
    assert result == expected
    assert type(result) is type(expected)
