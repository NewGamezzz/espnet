"""Tests for dialogue record emission (Task 5): TAC per-channel records and
the mono combined-stream record, byte-shape checked against the real SFT
schema documented in ``docs/bagpiper-findings.md``."""

from pathlib import Path

import pytest
from dataset.emit import (
    SYSTEM_PROMPT,
    emit_mono_record,
    emit_tac_records,
    is_tac_eligible,
)
from dataset.preprocessing.attributes import SpeakerAttrs
from dataset.preprocessing.audio import WindowAudio
from dataset.preprocessing.sssd import Turn
from dataset.preprocessing.windows import WindowRecord


def attrs(
    gender="female", pitch_band="medium", variability_band="flat", rate_band="moderate"
):
    return SpeakerAttrs(
        median_f0=180.0,
        f0_iqr=20.0,
        words_per_sec=3.0,
        pitch_band=pitch_band,
        variability_band=variability_band,
        rate_band=rate_band,
        gender=gender,
        gender_source="metadata",
    )


def window(turns, window_id="sess1_w00000", num_channels=2, t0=0.0, t1=10.0):
    return WindowRecord(
        window_id=window_id,
        session_id="sess1",
        audio_relpath="original/sess1_mixed.wav",
        num_channels=num_channels,
        sample_rate=48000,
        t0=t0,
        t1=t1,
        turns=tuple(turns),
    )


def window_audio(win, tmp_path):
    channel_paths = tuple(
        tmp_path / f"{win.window_id}_ch{c}.wav" for c in range(win.num_channels)
    )
    mix_path = tmp_path / f"{win.window_id}_mix.wav"
    return WindowAudio(
        window_id=win.window_id,
        channel_paths=channel_paths,
        mix_path=mix_path,
        channel_durations=tuple(win.duration for _ in channel_paths),
        mix_duration=win.duration,
    )


TWO_SPEAKER_TURNS = [
    Turn(channel=0, speaker="spk0", text="hello there", start=0.5, end=2.0),
    Turn(channel=1, speaker="spk1", text="hi how are you", start=2.5, end=4.0),
    Turn(channel=0, speaker="spk0", text="doing well thanks", start=4.5, end=6.0),
]

ONE_SPEAKER_TURNS = [
    Turn(channel=0, speaker="spk0", text="hello there", start=0.5, end=2.0),
    Turn(channel=0, speaker="spk0", text="how are you today", start=2.5, end=4.0),
]


class TestIsTacEligible:
    def test_two_active_channels_is_eligible(self):
        assert is_tac_eligible(window(TWO_SPEAKER_TURNS)) is True

    def test_one_active_channel_is_not_eligible(self):
        assert is_tac_eligible(window(ONE_SPEAKER_TURNS)) is False

    def test_multi_speaker_per_channel_raises(self):
        bad_turns = [
            Turn(channel=0, speaker="spk0", text="a", start=0.5, end=1.0),
            Turn(channel=0, speaker="spk_other", text="b", start=1.5, end=2.0),
        ]
        with pytest.raises(ValueError, match="channel 0"):
            is_tac_eligible(window(bad_turns))


class TestEmitTacRecords:
    def test_drops_windows_with_fewer_than_two_active_speakers(self, tmp_path):
        win = window(ONE_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        records = emit_tac_records(win, {"spk0": attrs()}, wa)
        assert records == []

    def test_one_record_per_active_channel(self, tmp_path):
        win = window(TWO_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {
            "spk0": attrs(gender="male"),
            "spk1": attrs(gender="female"),
        }
        records = emit_tac_records(win, attrs_by_speaker, wa)
        assert len(records) == 2
        assert {r["metadata"]["channel"] for r in records} == {0, 1}

    def test_schema_byte_shape(self, tmp_path):
        win = window(TWO_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {"spk0": attrs(), "spk1": attrs()}
        records = emit_tac_records(win, attrs_by_speaker, wa)
        rec = records[0]
        assert set(rec.keys()) == {"example_id", "messages", "metadata"}
        assert rec["example_id"] == f"sssd_tac_{win.window_id}_ch0"
        messages = rec["messages"]
        assert len(messages) == 4
        assert messages[0] == ["system", "text", SYSTEM_PROMPT]
        assert messages[1][0] == "user" and messages[1][1] == "text"
        assert isinstance(messages[1][2], str) and messages[1][2]
        assert messages[2][0] == "assistant" and messages[2][1] == "text"
        assert messages[2][2].startswith("<think>")
        assert messages[3] == ["assistant", "audio", str(wa.channel_paths[0])]
        assert Path(messages[3][2]).is_absolute()
        assert rec["metadata"] == {
            "conv_id": win.window_id,
            "channel": 0,
            "num_channels": win.num_channels,
            "speaker": "spk0",
            "t0": win.t0,
            "t1": win.t1,
        }

    def test_num_channels_reflects_window_not_hardcoded(self, tmp_path):
        turns = [
            Turn(channel=0, speaker="spk0", text="hello there", start=0.5, end=2.0),
            Turn(channel=2, speaker="spk2", text="hi there friend", start=2.5, end=4.0),
        ]
        win = window(turns, num_channels=3)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {"spk0": attrs(), "spk2": attrs()}
        records = emit_tac_records(win, attrs_by_speaker, wa)
        assert all(r["metadata"]["num_channels"] == 3 for r in records)
        assert {r["metadata"]["channel"] for r in records} == {0, 2}

    def test_script_is_channel_turns_joined_by_start_time(self, tmp_path):
        win = window(TWO_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {"spk0": attrs(), "spk1": attrs()}
        records = emit_tac_records(win, attrs_by_speaker, wa)
        ch0_record = next(r for r in records if r["metadata"]["channel"] == 0)
        caption = ch0_record["messages"][1][2]
        assert '"hello there doing well thanks"' in caption

    def test_conv_id_join_across_channel_records(self, tmp_path):
        win = window(TWO_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {"spk0": attrs(), "spk1": attrs()}
        records = emit_tac_records(win, attrs_by_speaker, wa)
        conv_ids = {r["metadata"]["conv_id"] for r in records}
        assert conv_ids == {win.window_id}
        example_ids = {r["example_id"] for r in records}
        assert example_ids == {
            f"sssd_tac_{win.window_id}_ch0",
            f"sssd_tac_{win.window_id}_ch1",
        }

    def test_multi_speaker_per_channel_raises(self, tmp_path):
        bad_turns = [
            Turn(channel=0, speaker="spk0", text="a b c", start=0.5, end=1.0),
            Turn(channel=0, speaker="spk_other", text="d e f", start=1.5, end=2.0),
            Turn(channel=1, speaker="spk1", text="g h i", start=2.5, end=3.0),
        ]
        win = window(bad_turns)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {"spk0": attrs(), "spk_other": attrs(), "spk1": attrs()}
        with pytest.raises(ValueError, match="channel 0"):
            emit_tac_records(win, attrs_by_speaker, wa)


class TestEmitMonoRecord:
    def test_schema_byte_shape(self, tmp_path):
        win = window(TWO_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {
            "spk0": attrs(gender="male"),
            "spk1": attrs(gender="female"),
        }
        rec = emit_mono_record(win, attrs_by_speaker, wa)
        assert set(rec.keys()) == {"example_id", "messages", "metadata"}
        assert rec["example_id"] == f"sssd_mono_{win.window_id}"
        messages = rec["messages"]
        assert len(messages) == 4
        assert messages[0] == ["system", "text", SYSTEM_PROMPT]
        assert messages[1][0] == "user" and messages[1][1] == "text"
        assert messages[2][0] == "assistant" and messages[2][1] == "text"
        assert messages[2][2].startswith("<think>")
        assert messages[3] == ["assistant", "audio", str(wa.mix_path)]
        assert Path(messages[3][2]).is_absolute()
        assert rec["metadata"] == {
            "conv_id": win.window_id,
            "variant": "mono",
            "speakers": ["spk0", "spk1"],
            "t0": win.t0,
            "t1": win.t1,
        }

    def test_kept_for_single_active_speaker(self, tmp_path):
        win = window(ONE_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        rec = emit_mono_record(win, {"spk0": attrs()}, wa)
        assert rec["metadata"]["speakers"] == ["spk0"]

    def test_speaker_labels_use_no_leading_article_template_path(self, tmp_path):
        """Regression guard: mono voice lines must render via the template
        path (no leading article stripped is fine, but no verbatim
        'A female speaker...' from a naive voice_description round-trip
        through the ``descriptions`` override either)."""
        win = window(TWO_SPEAKER_TURNS)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {
            "spk0": attrs(gender="male"),
            "spk1": attrs(gender="female"),
        }
        rec = emit_mono_record(win, attrs_by_speaker, wa)
        caption = rec["messages"][1][2]
        assert "Speaker 1: A " not in caption
        assert "Speaker 2: A " not in caption

    def test_multi_speaker_per_channel_raises(self, tmp_path):
        bad_turns = [
            Turn(channel=0, speaker="spk0", text="a b c", start=0.5, end=1.0),
            Turn(channel=0, speaker="spk_other", text="d e f", start=1.5, end=2.0),
        ]
        win = window(bad_turns)
        wa = window_audio(win, tmp_path)
        attrs_by_speaker = {"spk0": attrs(), "spk_other": attrs()}
        with pytest.raises(ValueError, match="channel 0"):
            emit_mono_record(win, attrs_by_speaker, wa)
