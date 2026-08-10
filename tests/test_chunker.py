from nugget_rag.chunker import split_sentences


def test_split_basic():
    text = "This is sentence one. This is sentence two. And three!"
    assert len(split_sentences(text)) == 3


def test_split_empty():
    assert split_sentences("") == []


def test_split_single():
    assert split_sentences("Only one sentence") == ["Only one sentence"]
