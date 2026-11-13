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


def test_split_keeps_decimals_intact():
    """#198: '0.5' / '88.3%' must not be cut at the decimal point."""
    text = "We add noise with a probability of 0.5 at a random SNR. It accounts for 88.3% of the variance."
    sentences = split_sentences(text)
    assert sentences == [
        "We add noise with a probability of 0.5 at a random SNR.",
        "It accounts for 88.3% of the variance.",
    ]


def test_split_keeps_inline_abbreviation_without_space():
    """'Fig.2' and 'v1.2.3' have no whitespace after the dot, so they stay whole."""
    sentences = split_sentences("See Fig.2 for details. Version v1.2.3 is used.")
    assert sentences == ["See Fig.2 for details.", "Version v1.2.3 is used."]


def test_split_japanese_with_decimal():
    """CJK terminators still split without whitespace; decimals inside are kept."""
    sentences = split_sentences("精度は0.95です。次の文です。")
    assert sentences == ["精度は0.95です。", "次の文です。"]


def test_split_keeps_abbreviations_intact():
    """#283: abbreviations / figure refs followed by a space must not end a sentence."""
    cases = [
        "Smith et al. 2020 showed gains.",
        "Dense retrievers, e.g. BM25 variants, are strong.",
        "Fig. 3 shows the latency.",
        "The bound follows from Eq. (2) directly.",
        "Ours beats dense vs. sparse baselines.",
        "That is, i.e. the same thing.",
        "See Sec. 4 and cf. Table 2.",
        "Results are in Table No. 5 below.",
    ]
    for case in cases:
        assert split_sentences(case) == [case]


def test_split_abbreviation_is_case_insensitive():
    assert split_sentences("fig. 3 shows it. FIG. 4 too.") == ["fig. 3 shows it.", "FIG. 4 too."]


def test_split_after_abbreviation_still_splits_real_boundary():
    text = "Smith et al. 2020 showed gains. Fig. 3 shows it. Done."
    assert split_sentences(text) == ["Smith et al. 2020 showed gains.", "Fig. 3 shows it.", "Done."]


def test_split_sentence_ending_in_non_abbreviation_word_splits():
    assert split_sentences("We evaluate on Fig data. Then we stop.") == [
        "We evaluate on Fig data.",
        "Then we stop.",
    ]


def test_split_no_as_answer_word_splits():
    assert split_sentences("The answer is no. Next we go.") == ["The answer is no.", "Next we go."]


def test_split_keeps_parenthesized_abbreviations_intact():
    assert split_sentences("Scores (e.g. BM25) vary. See [Fig. 3] for details.") == [
        "Scores (e.g. BM25) vary.",
        "See [Fig. 3] for details.",
    ]
