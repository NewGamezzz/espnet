"""Tests for ``local/mooncast_preflight.py``, the no-GPU input check.

The pre-flight exists to answer two questions before a GPU-second is spent:
how much of our script MoonCast's ``_clean_text`` rewrites, and whether any
dialogue's flat token sequence would outgrow the model's sequence length at
50 Hz.  These tests pin the accounting; the real cleaner and the real
tokenizer are injected, so nothing here needs their environment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "mooncast_preflight",
    Path(__file__).resolve().parents[1] / "local" / "mooncast_preflight.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def clean(text: str) -> str:
    """The parts of their ``_clean_text`` these tests exercise."""
    return text.replace("...", " ").replace(":", ",").replace("*", "").strip()


def encode(text: str) -> list[int]:
    """One token per character, so the counting stays checkable by hand."""
    return list(range(len(text)))


ROW = {
    "window_id": "d2",
    "role_mapping": {
        "0": {"ref_audio": "a.wav", "ref_text": "abc"},
        "1": {"ref_audio": "b.wav", "ref_text": "de"},
    },
    "dialogue": [
        {"role": "0", "text": "abcd"},
        {"role": "1", "text": "ef"},
    ],
}


def _check(rows, ceiling=40000, targets=None):
    return mod.check_rows(
        rows, clean, encode, lambda path: 2.0, ceiling, target_secs=targets
    )


class TestEstimate:
    def test_all_four_kinds_of_token_are_counted(self):
        estimate = mod.estimate_context_tokens([2.0, 2.0], 10.0, [4, 2], 2)
        assert estimate["prompt_audio_tokens"] == 200
        assert estimate["target_audio_tokens"] == 500
        # (4 + 5) + (2 + 5)
        assert estimate["text_tokens"] == 16
        # 9 tokens of framing for each of 2 prompts and 2 turns
        assert estimate["framing_tokens"] == 36
        assert estimate["tokens"] == 200 + 500 + 16 + 36

    def test_audio_is_counted_at_fifty_hertz(self):
        # Four times FireRedTTS-2's rate, which is why this is checked at
        # all rather than assumed to be slack.
        assert mod.FRAMES_PER_SEC == 50
        estimate = mod.estimate_context_tokens([1.0], 1.0, [], 0)
        assert estimate["prompt_audio_tokens"] == 50
        assert estimate["target_audio_tokens"] == 50

    def test_a_dialogue_with_no_reference_duration_counts_no_generated_audio(self):
        estimate = mod.estimate_context_tokens([1.0], 0.0, [3], 1)
        assert estimate["target_audio_tokens"] == 0


class TestCheckRows:
    def test_an_unaltered_row_is_not_reported_as_cleaned(self):
        report = _check([ROW])
        assert report["cleaned_rows"] == []
        assert report["over_ceiling"] == []
        assert report["rows"][0]["window_id"] == "d2"
        assert report["rows"][0]["num_turns"] == 2

    def test_a_row_their_cleaner_rewrites_is_reported(self):
        # MoonCast is the first re-run baseline whose input text differs
        # from the reference transcript, so this has to be counted.
        row = dict(ROW, dialogue=[{"role": "0", "text": "well: yes"}])
        assert _check([row])["cleaned_rows"] == ["d2"]

    def test_a_row_cleaned_only_in_a_reference_text_is_reported(self):
        row = dict(
            ROW,
            role_mapping={
                "0": {"ref_audio": "a.wav", "ref_text": "no*pe"},
                "1": {"ref_audio": "b.wav", "ref_text": "de"},
            },
        )
        assert _check([row])["cleaned_rows"] == ["d2"]

    def test_token_counts_come_from_the_cleaned_text(self):
        row = dict(ROW, dialogue=[{"role": "0", "text": "a...b"}])
        # "a b" is 3 characters after cleaning, plus 5 framing, plus the
        # two reference texts (3 + 5) and (2 + 5).
        assert _check([row])["rows"][0]["text_tokens"] == 8 + 8 + 7

    def test_a_row_over_the_ceiling_is_reported(self):
        report = _check([ROW], ceiling=10)
        assert report["over_ceiling"] == ["d2"]

    def test_the_reference_duration_feeds_the_estimate(self):
        without = _check([ROW])["rows"][0]["tokens"]
        with_target = _check([ROW], targets={"d2": 20.0})["rows"][0]["tokens"]
        assert with_target - without == 1000
