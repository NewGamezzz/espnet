import pytest
from src.text.lemas_phonemizer import (
    LANG_TOKEN,
    SPACE_TOKEN,
    SPK_TOKEN,
    LEMASPhonemizer,
    lang_tag,
    special_tokens,
)

pytest.importorskip("phonemizer")


def _has_espeak():
    try:
        from phonemizer.backend import EspeakBackend

        return EspeakBackend.is_available()
    except Exception:
        return False


def test_tokens():
    assert lang_tag("de") == "<de>"
    assert special_tokens()[:2] == [SPK_TOKEN, LANG_TOKEN]
    assert "<zh>" in special_tokens() and SPACE_TOKEN not in special_tokens()


def test_zh_pinyin_initial_final():
    ph = LEMASPhonemizer(langs=["zh"]).phonemize("你好", "zh")
    assert ph == ["n", "i3", "h", "ao3"]


@pytest.mark.skipif(not _has_espeak(), reason="espeak-ng not available")
def test_de_per_phone_with_space_and_stress():
    ph = LEMASPhonemizer(langs=["de"]).phonemize("Guten Morgen, Welt.", "de")
    assert SPACE_TOKEN in ph
    assert all(len(t) <= 4 for t in ph if t != SPACE_TOKEN)  # per-phone tokens
    assert any("ˈ" in t for t in ph)  # stress kept
    assert "," in ph and "." in ph  # punctuation kept
    assert not any("(" in t for t in ph)  # no espeak language flags


@pytest.mark.skipif(not _has_espeak(), reason="espeak-ng not available")
def test_phonemize_words_returns_one_list_per_word():
    out = LEMASPhonemizer(langs=["ru"]).phonemize_words(["привет", "мир"], "ru")
    assert len(out) == 2 and all(out)


def test_unknown_lang_raises():
    with pytest.raises(KeyError):
        LEMASPhonemizer(langs=["de"]).phonemize("x", "xx")


def test_unicode_punctuation_splits_off_anywhere():
    from src.text.lemas_phonemizer import _split_punct

    assert _split_punct("¿k") == ["¿", "k"]
    assert _split_punct("t»") == ["t", "»"]
    assert _split_punct("oː“") == ["oː", "“"]
    assert _split_punct("ˈiː.dʒ") == ["ˈiː", ".", "dʒ"]
    assert _split_punct("，") == ["，"]
    assert _split_punct("[]") == ["[", "]"]


def test_espeak_phone_symbols_with_hyphen_or_caret_survive():
    from src.text.lemas_phonemizer import _split_punct

    # French elidable schwa, Vietnamese vowel with tone, Russian soft-sign vowel
    assert _split_punct("ə-") == ["ə-"]
    assert _split_punct("ˈe-2") == ["ˈe-2"]
    assert _split_punct("ɪ^") == ["ɪ^"]


def test_spanish_inverted_marks_become_tokens():
    ph = LEMASPhonemizer(["es"])
    out = ph.phonemize("¿Cómo estás? ¡Hola!", "es")
    assert out[0] == "¿" and "¡" in out and "?" in out and "!" in out
    assert all(not (len(t) > 1 and t[0] in "¿¡") for t in out)
