"""Fisher manifest parsing, text cleaning, and A/B -> stereo merge machinery."""

import gzip
import json
import string
import subprocess
from pathlib import Path

import pytest

from egs3.conversational.tts.dataset.preprocessing import fisher
from egs3.conversational.tts.dataset.preprocessing.sssd import Supervision

from .conftest import REPO_ROOT, write_flac  # noqa: F401  (sys.path setup)


def sup(text, start=0.0, duration=1.0, channel=0, speaker="s0"):
    return Supervision(
        id=f"u-{start}",
        recording_id="fe_03_00001",
        channel=channel,
        start=start,
        duration=duration,
        text=text,
        speaker=speaker,
    )


def test_clean_keeps_plain_text():
    res = fisher.clean_fisher_text("and i generally prefer")
    assert res == fisher.CleanResult("and i generally prefer", False)


def test_clean_strips_event_tags_keeping_words():
    res = fisher.clean_fisher_text("well [laughter] you know [noise] right")
    assert res == fisher.CleanResult("well you know right", False)


def test_clean_unwraps_unclear_markers():
    res = fisher.clean_fisher_text("i think (( yeah maybe )) so")
    assert res == fisher.CleanResult("i think yeah maybe so", False)


def test_clean_maps_underscore_to_space():
    res = fisher.clean_fisher_text("watching t._v. tonight")
    assert res == fisher.CleanResult("watching t. v. tonight", False)


def test_clean_tag_only_utterance_is_benign_empty():
    res = fisher.clean_fisher_text("[laughter]")
    assert res == fisher.CleanResult("", False)
    res = fisher.clean_fisher_text("[sigh] [lipsmack]")
    assert res == fisher.CleanResult("", False)


def test_clean_empty_unclear_is_unintelligible():
    assert fisher.clean_fisher_text("(( ))").unintelligible
    assert fisher.clean_fisher_text("(( [noise] ))").unintelligible


def test_clean_foreign_and_unrepresentable_are_unintelligible():
    assert fisher.clean_fisher_text("<german (( ja wohl )) >").unintelligible
    assert fisher.clean_fisher_text("call me at 1 800").unintelligible
    assert fisher.clean_fisher_text("a. t. & t.").unintelligible
    assert fisher.clean_fisher_text("uh *huh*").unintelligible


def test_clean_double_bracket_skip_tag():
    # "[[skip]" appears in the corpus; the tag regex must consume it whole.
    res = fisher.clean_fisher_text("[[skip] and then")
    assert res == fisher.CleanResult("and then", False)


def test_clean_supervisions_partitions_and_collects_spans():
    sups = [
        sup("hello there", start=0.0, duration=2.0),
        sup("[laughter]", start=2.0, duration=0.5),          # benign drop
        sup("(( ))", start=3.0, duration=1.0),                # span
        sup("so [noise] anyway", start=5.0, duration=2.0),    # kept, cleaned
        sup("call 911 now", start=8.0, duration=1.0),         # span (digits)
    ]
    kept, spans, n_benign = fisher.clean_fisher_supervisions(sups)
    assert [s.text for s in kept] == ["hello there", "so anyway"]
    # cleaned supervisions keep their timing/channel/speaker fields
    assert kept[1].start == 5.0 and kept[1].duration == 2.0
    assert spans == [(3.0, 4.0), (8.0, 9.0)]
    assert n_benign == 1
