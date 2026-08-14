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


def test_en_repetition_uses_length_four():
    """Upstream uses length=4 for EN, length=2 for ZH."""
    assert filters.repetition_found("a b c d a b c d", length=4) is True
    assert filters.repetition_found("a b c d e f g h", length=4) is False
    # A 2-gram repeat must NOT trip the EN rule.
    assert filters.repetition_found("a b x y a b q r", length=4) is False


def test_zh_repetition_uses_length_two():
    assert filters.repetition_found("你好 世界 你好 世界", length=2) is True


def test_zh_punctuation_is_normalized():
    assert filters.normalize_text("你好,世界!好吗?", "ZH") == "你好，世界！好吗？"


def test_en_text_is_not_punctuation_normalized():
    assert filters.normalize_text("hello, world!", "EN") == "hello, world!"


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
    """D3: filter_text.py rules must not fire unless explicitly enabled."""
    rec = _rec(speaker="EN_B00000_S00000", text="ooooooooooooooo dear")
    assert filters.keep_utterance(rec, "EN", 0.3, 30.0, strict=False)[0] is True
    assert filters.keep_utterance(rec, "EN", 0.3, 30.0, strict=True) == (
        False,
        "strict",
    )
