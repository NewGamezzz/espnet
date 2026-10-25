"""Stock F5-TTS front-end (characters, pinyin for zh) behind the builder interface."""

import pytest
from src.text.char_tokenizer import TEXT_UNITS, F5CharTokenizer, text_frontend

pytest.importorskip("rjieba")
pytest.importorskip("pypinyin")


def test_latin_text_is_character_level_with_space_tokens():
    assert F5CharTokenizer().phonemize("hi du", "de") == ["h", "i", "<space>", "d", "u"]


def test_chinese_is_pinyin_syllables_with_tone_numbers():
    # F5's Emilia_ZH_EN_pinyin scheme puts a space before every syllable
    assert F5CharTokenizer().phonemize("中文", "zh") == [
        "<space>",
        "zhong1",
        "<space>",
        "wen2",
    ]


def test_cyrillic_and_vietnamese_stay_character_level():
    assert F5CharTokenizer().phonemize("да", "ru") == ["д", "а"]
    assert F5CharTokenizer().phonemize("ươ", "vi") == ["ư", "ơ"]


def test_whitespace_of_any_kind_becomes_space_tokens():
    # the manifest is tab-separated and ManifestColumns.load raises on a wrong
    # column count, so a tab or a no-break space must never survive as a token;
    # every whitespace character is one token, as F5 maps them
    toks = F5CharTokenizer().phonemize("a  b\tc\u00a0d", "en")
    assert toks == ["a", "<space>", "<space>", "b", "<space>", "c", "<space>", "d"]
    for t in toks:
        assert t and "|" not in t and not any(ch.isspace() for ch in t)


def test_frontend_dispatch():
    assert TEXT_UNITS == ("phones", "chars")
    assert type(text_frontend("chars")).__name__ == "F5CharTokenizer"
    with pytest.raises(ValueError):
        text_frontend("bytes")


def test_frontend_factory_survives_the_process_pool_initializer():
    # the builder hands functools.partial(text_frontend, units) to a
    # ProcessPoolExecutor initializer: it must pickle (regression guard, the
    # fake-mirror build tests run with n_workers=1 and never pickle it)
    import functools
    import pickle

    f = pickle.loads(pickle.dumps(functools.partial(text_frontend, "chars")))
    assert f().phonemize("hi", "en") == ["h", "i"]


def test_pipe_is_a_character_token_like_stock_f5():
    # YODAS captions carry "|"; stock F5 keeps every character, and "|" only
    # delimits the split_targets column, which the builder guards itself
    assert F5CharTokenizer().phonemize("a|b", "en") == ["a", "|", "b"]


def test_whitespace_runs_are_kept_like_stock_f5():
    # F5 emits a space before and after a Chinese segment, so a space-joined
    # boundary yields two " " tokens; stock F5 maps both, so S0a keeps both
    assert F5CharTokenizer().phonemize("hello 世界", "zh") == [
        "h", "e", "l", "l", "o", "<space>", "<space>", "shi4", "<space>", "jie4",
    ]
