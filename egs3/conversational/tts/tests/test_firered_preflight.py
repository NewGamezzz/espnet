"""Tests for ``local/firered_preflight.py``, the no-GPU gate.

Two of FireRedTTS-2's limits can end a row: they re-segment any turn past 80
English words, and ``generate`` RAISES once the context outgrows
``max_seq_len - max_generation_len``.  Both are knowable from the input file
plus their own splitter, so they are knowable before a GPU-second is spent.

Their splitter is injected here rather than imported: on Delta the real
``process_text_list`` runs, and these tests cover our accounting around it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "firered_preflight",
    Path(__file__).resolve().parents[1] / "local" / "firered_preflight.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def identity_splitter(text_list):
    return list(text_list)


def doubling_splitter(text_list):
    """Their 80-word rule, exaggerated: every turn becomes two."""
    return [t for text in text_list for t in (text, text)]


ROW = {
    "window_id": "d2",
    "text_list": ["[S1] hello there", "[S2] hi"],
    "prompt_wav_list": ["/p/a.wav", "/p/b.wav"],
    "prompt_text_list": ["[S1] abc", "[S2] de"],
}


def durations(mapping, default=4.0):
    return lambda path: mapping.get(str(path), default)


class TestSplitAccounting:
    def test_an_unsplit_row_is_reported_as_such(self):
        report = mod.check_rows([ROW], identity_splitter, durations({}))
        assert report["rows"][0]["num_turns_in"] == 2
        assert report["rows"][0]["num_turns_after"] == 2
        assert report["split_rows"] == []

    def test_a_split_row_is_named(self):
        # A system that silently re-segments our script is a fact any
        # writeup has to state, so it is counted rather than discovered.
        report = mod.check_rows([ROW], doubling_splitter, durations({}))
        assert report["rows"][0]["num_turns_after"] == 4
        assert report["split_rows"] == ["d2"]

    def test_a_lost_speaker_tag_is_an_error(self):
        def stripping(text_list):
            return [text[4:] for text in text_list]

        with pytest.raises(ValueError, match="tag"):
            mod.check_rows([ROW], stripping, durations({}))


class TestContextEstimate:
    def test_prompt_and_target_audio_both_count(self):
        # Context is prompts + every turn generated so far, all at 12.5 Hz.
        est = mod.estimate_context_frames(
            prompt_secs=[8.0, 8.0], target_sec=24.0, texts=[]
        )
        assert est["prompt_frames"] == 200
        assert est["target_frames"] == 300
        assert est["frames"] == 500

    def test_text_tokens_are_included(self):
        # They are frames of the same sequence: dropping them would make
        # the estimate optimistic in exactly the wrong direction.
        with_text = mod.estimate_context_frames([], 0.0, ["[S1] " + "a " * 100])
        assert with_text["text_frames"] > 0
        assert with_text["frames"] == with_text["text_frames"]

    def test_a_row_past_their_ceiling_is_flagged(self):
        long_row = {**ROW, "text_list": ["[S1] hi"]}
        report = mod.check_rows(
            [long_row],
            identity_splitter,
            durations({}, default=0.0),
            target_secs={"d2": 400.0},
        )
        # 400 s of audio is 5000 frames against a ceiling of 2725.
        assert report["over_ceiling"] == ["d2"]
        assert report["rows"][0]["est_context_frames"] > mod.CONTEXT_CEILING

    def test_a_normal_row_is_not_flagged(self):
        report = mod.check_rows(
            [ROW],
            identity_splitter,
            durations({}, default=8.0),
            target_secs={"d2": 24.0},
        )
        assert report["over_ceiling"] == []

    def test_the_ceiling_is_their_two_constants(self):
        # max_seq_len 3100 minus max_generation_len (30_000 ms / 80 ms).
        assert mod.CONTEXT_CEILING == 3100 - 375
