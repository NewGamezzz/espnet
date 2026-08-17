"""Upstream-faithful Emilia filters."""

import pytest

from egs3.emilia.tts.dataset import filters


def _rec(**over):
    base = {
        "id": "EN_B00013_S00913_W000004",
        "text": "a normal english sentence about nothing at all",
        "duration": 5.0,
        "speaker": "EN_B00013_S00913",
        "language": "en",
    }
    base.update(over)
    return base


def test_blocklist_matches_on_speaker_not_id():
    """Regression for the staged prep's inert blocklist (spec 2.2).

    The blocklists hold speaker ids; obj["id"] carries an extra _W field, so
    matching on "id" can never fire.
    """
    assert "EN_B00013_S00913" in filters.OUT_EN
    keep, reason = filters.keep_utterance(_rec(), "EN", 0.3, 30.0)
    assert keep is False
    assert reason == "blocklist"


def test_non_blocklisted_speaker_is_kept():
    keep, reason = filters.keep_utterance(
        _rec(speaker="EN_B00000_S00000", id="EN_B00000_S00000_W000000"),
        "EN",
        0.3,
        30.0,
    )
    assert keep is True
    assert reason == ""


@pytest.mark.parametrize("bad", ["ا", "い", "て"])
def test_en_cross_language_chars_rejected(bad):
    keep, reason = filters.keep_utterance(
        _rec(speaker="EN_B00000_S00000", text=f"hello {bad} world"),
        "EN",
        0.3,
        30.0,
    )
    assert (keep, reason) == (False, "charfilter")


def _upstream_repetition_found(text, length=2, tolerance=10):
    """Verbatim f5_tts.model.utils.repetition_found, as the reference oracle.

    Duplicated here on purpose: F5-TTS is not an import dependency of this
    recipe, and the point of the tests below is to pin our port against
    upstream's actual algorithm rather than against our own restatement of
    it. An earlier version of dataset/filters.py claimed to be a port and
    was not, and the tests that accompanied it encoded the divergence as
    expected behaviour instead of catching it.
    """
    pattern_count = {}
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
    return any(count > tolerance for count in pattern_count.values())


def test_repetition_counts_characters_not_words():
    """The n-grams are character slices. This is what breaks ZH if got wrong.

    Emilia's Chinese text carries no spaces, so a word-based implementation
    collapses to a single token and the filter silently never fires for the
    entire Chinese half of the corpus.
    """
    assert filters.repetition_found("啊" * 15, length=2) is True
    assert filters.repetition_found("今天天气很好我们一起去公园散步吧", length=2) is False


def test_repetition_tolerance_is_ten_not_one():
    """`count > tolerance` with tolerance=10, not "appears twice".

    Ordinary English repeats short phrases; upstream keeps it. An
    appears-twice rule rejects this sentence, which is a large and silent
    over-filtering of the English half.
    """
    text = "I went to the store and I went to the store again yesterday."
    assert filters.repetition_found(text, length=4) is False
    assert _upstream_repetition_found(text, length=4) is False

    # 11 occurrences of a 2-gram trips it; 10 does not.
    assert filters.repetition_found("ab" * 10, length=2) is False
    assert filters.repetition_found("ab" * 11, length=2) is True


def test_repetition_matches_upstream_on_random_text():
    """Differential test against the upstream oracle.

    Case-by-case assertions are what let the original divergence through:
    each one was individually true of the wrong implementation. Random
    differential testing over the alphabet the corpus actually uses is what
    catches a wrong algorithm rather than a wrong example.
    """
    import random
    import string

    rng = random.Random(0)
    alphabet = string.ascii_lowercase + " ,.!?" + "你好今天气很我们去公园散步吧啊"
    for _ in range(4000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        for length in (2, 4):
            assert filters.repetition_found(text, length=length) is (
                _upstream_repetition_found(text, length=length)
            ), (length, text)


def test_en_uses_length_four_and_zh_length_two():
    """The per-language lengths match upstream's call sites:
    `repetition_found(text, length=4)` for EN, `repetition_found(text)` for ZH."""
    assert filters._REPETITION_LENGTH_BY_LANG["EN"] == 4
    assert filters._REPETITION_LENGTH_BY_LANG["ZH"] == 2


def test_zh_punctuation_is_normalized():
    assert filters.normalize_text("你好,世界!好吗?", "ZH") == "你好，世界！好吗？"


def test_en_text_is_not_punctuation_normalized():
    assert filters.normalize_text("hello, world!", "EN") == "hello, world!"


def test_embedded_newlines_and_tabs_are_stripped():
    """dataset/builder.py writes normalize_text's output as the last,
    unquoted field of a tab-separated manifest row: a literal '\\n', '\\r'
    or '\\t' would otherwise split or shift that row, corrupting the
    manifest at 37M-row scale."""
    assert filters.normalize_text("hello\nworld", "EN") == "hello world"
    assert filters.normalize_text("hello\r\nworld", "EN") == "hello  world"
    assert filters.normalize_text("hello\tworld", "EN") == "hello world"
    assert "\n" not in filters.normalize_text("a\nb\tc\rd", "ZH")
    assert "\t" not in filters.normalize_text("a\nb\tc\rd", "ZH")
    assert "\r" not in filters.normalize_text("a\nb\tc\rd", "ZH")


@pytest.mark.parametrize(
    "dur,expected",
    [
        (0.29, False),
        (0.3, True),
        (5.0, True),
        (30.0, True),
        (30.1, False),
    ],
)
def test_duration_bounds_are_f5_emilia_bounds(dur, expected):
    keep, reason = filters.keep_utterance(
        _rec(speaker="EN_B00000_S00000", duration=dur),
        "EN",
        0.3,
        30.0,
    )
    assert keep is expected
    if not expected:
        assert reason == "duration"


def test_strict_filters_are_off_by_default():
    """D3: filter_text.py rules must not fire unless explicitly enabled.

    The example must trip ONLY the strict rules. It used to be
    "ooooooooooooooo dear", which stopped isolating them once
    repetition_found was corrected to upstream's algorithm: the 15-character
    run of "o" contains 12 occurrences of the 4-gram "oooo", past
    tolerance=10, so upstream's own repetition rule now rejects it too and
    the test could no longer tell the two filters apart. A 15-digit run
    trips _LONG_DIGITS while its 4-grams repeat at most twice.
    """
    rec = _rec(speaker="EN_B00000_S00000", text="123456789012345 dear")
    assert filters.keep_utterance(rec, "EN", 0.3, 30.0, strict=False)[0] is True
    assert filters.keep_utterance(rec, "EN", 0.3, 30.0, strict=True) == (
        False,
        "strict",
    )


def test_zh_blocklist_rejects_through_keep_utterance():
    """OUT_ZH membership rejects through keep_utterance itself, not just as
    a standalone frozenset -- the ZH-side counterpart to
    test_blocklist_matches_on_speaker_not_id (EN-side)."""
    speaker = "ZH_B00041_S06226"
    assert speaker in filters.OUT_ZH
    keep, reason = filters.keep_utterance(
        _rec(speaker=speaker, id=f"{speaker}_W000000", text="你好世界", language="zh"),
        "ZH",
        0.3,
        30.0,
    )
    assert (keep, reason) == (False, "blocklist")


@pytest.mark.parametrize("bad", ["い", "て"])
def test_zh_char_filters_reject_through_keep_utterance(bad):
    """ZH_CHAR_FILTERS rejects through keep_utterance itself, not just
    through `ch in text` in isolation -- the ZH-side counterpart to
    test_en_cross_language_chars_rejected (EN-side)."""
    keep, reason = filters.keep_utterance(
        _rec(
            speaker="ZH_B00000_S00000",
            id="ZH_B00000_S00000_W000000",
            text=f"你好{bad}世界",
            language="zh",
        ),
        "ZH",
        0.3,
        30.0,
    )
    assert (keep, reason) == (False, "charfilter")


def test_unknown_lang_raises():
    """IMPORTANT 4: keep_utterance used to silently fall back to the ZH
    rule for any lang that wasn't the literal string "EN" -- including a
    typo or a genuinely new third language, since dataset/config.yaml
    exposes `langs` as a config knob with no validation against these
    tables. A dict lookup must raise instead of silently guessing."""
    with pytest.raises(ValueError, match="FR"):
        filters.keep_utterance(
            _rec(speaker="EN_B00000_S00000", language="fr"),
            "FR",
            0.3,
            30.0,
        )


def test_unknown_lang_raises_before_any_language_specific_work():
    """The blocklist lookup is the first per-language dispatch in
    keep_utterance, so an unknown lang must raise there rather than
    silently falling through to a later stage with the wrong table."""
    with pytest.raises(ValueError):
        filters._lookup_by_lang(filters._BLOCKLIST_BY_LANG, "FR", "blocklist")
    with pytest.raises(ValueError):
        filters._lookup_by_lang(filters._CHAR_FILTERS_BY_LANG, "FR", "char filters")
    with pytest.raises(ValueError):
        filters._lookup_by_lang(
            filters._REPETITION_LENGTH_BY_LANG, "FR", "repetition length"
        )


def test_normalize_text_preserves_leading_and_trailing_spaces():
    """Upstream stores obj["text"] verbatim; we must not strip it.

    100% of Emilia EN records carry a leading space, and F5's vocab tokenizes
    a space as a real token (index 0), so stripping changed the conditioning
    sequence on every English utterance. The official F5TTS_Base checkpoint
    was trained with the space present, and D7 keeps this recipe token-list
    compatible with it.
    """
    raw = " You can help my mother and you- No. "
    assert filters.normalize_text(raw, "EN") == raw

    # ZH still gets punctuation widened, but the leading space survives.
    assert filters.normalize_text(" 你好,世界!", "ZH") == " 你好，世界！"


def test_normalize_text_still_kills_row_breaking_whitespace():
    """Newlines/CR/tabs must go: builder.py writes this as the last unquoted
    field of a TSV row, so a literal one would split or shift the row. Spaces
    are TSV-safe and are deliberately kept."""
    out = filters.normalize_text("a\nb\tc\rd", "EN")
    assert out == "a b c d"
    assert "\n" not in out and "\t" not in out and "\r" not in out


def test_filters_see_the_raw_unstripped_text():
    """keep_utterance passes record["text"] through untouched, as upstream does.

    Verified to change no decision: 0 flips over 215,806 real records
    (job 43760108), since one boundary character cannot move a character
    n-gram count past tolerance=10.
    """
    import inspect

    src = inspect.getsource(filters.keep_utterance)
    assert 'record["text"]\n' in src or 'text = record["text"]' in src
    assert '.strip()' not in src
