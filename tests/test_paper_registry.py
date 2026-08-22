from nugget_rag.paper_registry import ARXIV_MAP, PAPER_ID_TO_ARXIV


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
