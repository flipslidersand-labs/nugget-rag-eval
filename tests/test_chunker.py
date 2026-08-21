from nugget_rag.chunker import split_sentences


def test_split_basic():
    text = "This is sentence one. This is sentence two. And three!"
    assert len(split_sentences(text)) == 3


def test_split_empty():
    assert split_sentences("") == []


def test_split_single():
    assert split_sentences("Only one sentence") == ["Only one sentence"]


def test_split_japanese():
    """Test CJK sentence splitting with Japanese punctuation."""
    text = "これは最初の文です。これは2番目の文です。3番目です！"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "これは最初の文です。"
    assert sentences[1] == "これは2番目の文です。"
    assert sentences[2] == "3番目です！"


def test_split_mixed_japanese_english():
    """Test mixed Japanese and English sentences."""
    text = "Hello world。これは日本語です。And English again!"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world。"
    assert sentences[1] == "これは日本語です。"
    assert sentences[2] == "And English again!"


def test_split_japanese_question():
    """Test Japanese question mark."""
    text = "質問です？答えはここ。"
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "質問です？"
    assert sentences[1] == "答えはここ。"
