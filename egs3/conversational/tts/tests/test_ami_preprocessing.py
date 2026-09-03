"""AMI ingestion: NXT parsing, text normalization, word-run supervisions,
headset transcode.  Fabricated XML/wav fixtures, no corpus access."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.dataset.preprocessing.ami import (
    TEST_MEETINGS,
    Participant,
    Word,
    headset_paths,
    load_ami_recordings,
    load_meetings,
    load_words,
    normalize_ami_text,
    transcode_meeting,
    words_to_supervisions,
)

NITE = 'xmlns:nite="http://nite.sourceforge.net/"'

MEETINGS_XML = f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root {NITE} nite:id="meetings">
  <meeting nite:id="meet_49" type="scenario" observation="ES2004a" duration="1186">
    <speaker nite:id="ES2004a_2" channel="1" nxt_agent="B" global_name="FEE013" role="PM"/>
    <speaker nite:id="ES2004a_3" channel="2" nxt_agent="C" global_name="MEE014" role="ID"/>
    <speaker nite:id="ES2004a_1" channel="0" nxt_agent="A" global_name="MEO015" role="UI"/>
    <speaker nite:id="ES2004a_4" channel="3" nxt_agent="D" global_name="FEE016" role="ME"/>
  </meeting>
</nite:root>
"""

WORDS_XML = f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root {NITE} nite:id="ES2004a.A.words">
  <w nite:id="w0" starttime="0.37" endtime="0.95">Hmm</w>
  <w nite:id="w1" starttime="0.95" endtime="1.53">hmm</w>
  <w nite:id="w2" starttime="1.53" endtime="1.53" punc="true">.</w>
  <vocalsound nite:id="v0" starttime="2.0" endtime="2.4" type="laugh"/>
  <w nite:id="w3" starttime="17.88" endtime="18.15">Yeah</w>
  <w nite:id="w4" starttime="18.15" endtime="18.15" punc="true">.</w>
  <w nite:id="w5" starttime="18.20" endtime="18.60">the</w>
  <w nite:id="w6" starttime="18.60" endtime="19.10">L_C_D</w>
  <w nite:id="w7" starttime="19.10" endtime="19.50">screen</w>
  <w nite:id="w8" starttime="19.50" endtime="19.50" punc="true">?</w>
  <gap nite:id="g0" starttime="19.6" endtime="19.9"/>
  <w nite:id="w9">untimed</w>
</nite:root>
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="ISO-8859-1")
    return p


class TestMeetings:
    def test_partition_list(self):
        assert len(TEST_MEETINGS) == 24
        assert TEST_MEETINGS[0] == "EN2002a"
        series = {"ES2004", "ES2014", "IS1009", "TS3003", "TS3007", "EN2002"}
        assert all(m[:-1] in series for m in TEST_MEETINGS)
        assert all(m[-1] in "abcd" for m in TEST_MEETINGS)

    def test_load_meetings_maps_agent_to_channel_and_speaker(self, tmp_path):
        parts = load_meetings(_write(tmp_path, "meetings.xml", MEETINGS_XML))
        assert list(parts) == ["ES2004a"]
        got = sorted(parts["ES2004a"], key=lambda p: p.channel)
        assert got == [
            Participant("A", 0, "MEO015"),
            Participant("B", 1, "FEE013"),
            Participant("C", 2, "MEE014"),
            Participant("D", 3, "FEE016"),
        ]

    def test_load_meetings_rejects_duplicate_channel_when_required(self, tmp_path):
        bad = MEETINGS_XML.replace('channel="3"', 'channel="2"')
        path = _write(tmp_path, "meetings.xml", bad)
        with pytest.raises(ValueError, match="ES2004a.*channel"):
            load_meetings(path, require=["ES2004a"])
        # the corpus file carries 3-participant meetings we never use: no
        # validation unless required
        assert "ES2004a" in load_meetings(path)

    def test_three_participant_meeting_is_accepted(self, tmp_path):
        three = MEETINGS_XML.replace(
            '<speaker nite:id="ES2004a_1" channel="0" nxt_agent="A" global_name="MEO015" role="UI"/>',
            "",
        )
        parts = load_meetings(_write(tmp_path, "meetings.xml", three), require=["ES2004a"])
        assert sorted(p.channel for p in parts["ES2004a"]) == [1, 2, 3]

    def test_complete_participants_synthesizes_agent_with_words(self, tmp_path):
        from egs3.conversational.tts.dataset.preprocessing.ami import complete_participants

        three = MEETINGS_XML.replace(
            '<speaker nite:id="ES2004a_1" channel="0" nxt_agent="A" global_name="MEO015" role="UI"/>',
            "",
        )
        parts = load_meetings(_write(tmp_path, "meetings.xml", three), require=["ES2004a"])["ES2004a"]
        words_dir = tmp_path / "words"
        words_dir.mkdir()
        (words_dir / "ES2004a.A.words.xml").write_text(WORDS_XML, encoding="ISO-8859-1")
        full, added = complete_participants("ES2004a", parts, words_dir)
        assert [p.channel for p in full] == [0, 1, 2, 3]
        assert added == [Participant("A", 0, "ES2004a.A")]
        # no words file -> nothing synthesized
        full2, added2 = complete_participants("ES2004a", parts, tmp_path / "nowords")
        assert added2 == [] and [p.channel for p in full2] == [1, 2, 3]

    def test_load_meetings_requires_presence(self, tmp_path):
        path = _write(tmp_path, "meetings.xml", MEETINGS_XML)
        with pytest.raises(KeyError, match="IN1001"):
            load_meetings(path, require=["IN1001"])


class TestWords:
    def test_load_words_keeps_timed_words_and_punctuation_only(self, tmp_path):
        words = load_words(_write(tmp_path, "w.xml", WORDS_XML))
        assert [w.text for w in words] == [
            "Hmm", "hmm", ".", "Yeah", ".", "the", "L_C_D", "screen", "?",
        ]
        assert words[2] == Word(1.53, 1.53, ".", True)
        assert all(w.end >= w.start for w in words)

    def test_load_words_rejects_non_monotone(self, tmp_path):
        bad = WORDS_XML.replace('starttime="17.88"', 'starttime="0.10"')
        with pytest.raises(ValueError, match="monoton"):
            load_words(_write(tmp_path, "w.xml", bad))


class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Hmm hmm .", "Hmm hmm."),
            ("the L_C_D screen ?", "the L C D screen?"),
            ("OKAY", "okay"),  # all-caps words are lowercased, never spelled
            ("I'm  fine ,  thanks", "I'm fine, thanks"),
            ("Mm-hmm .", "Mm-hmm."),
        ],
    )
    def test_cases(self, raw, expected):
        assert normalize_ami_text(raw) == expected


class TestSupervisions:
    def test_word_runs_split_at_gap(self, tmp_path):
        words = load_words(_write(tmp_path, "w.xml", WORDS_XML))
        sups = words_to_supervisions(
            words, meeting_id="ES2004a", channel=0, speaker="MEO015", utterance_gap=1.0
        )
        assert [s.text for s in sups] == ["Hmm hmm.", "Yeah. the L C D screen?"]
        assert sups[0].start == 0.37 and round(sups[0].end, 2) == 1.53
        assert sups[1].start == 17.88 and round(sups[1].end, 2) == 19.5
        assert sups[0].id == "ES2004a.A.u0000" and sups[1].id == "ES2004a.A.u0001"
        assert sups[0].recording_id == "ES2004a"
        assert sups[0].channel == 0 and sups[0].speaker == "MEO015"

    def test_punctuation_only_run_is_dropped(self, tmp_path):
        xml = f"""<?xml version="1.0"?>
<nite:root {NITE}>
  <w nite:id="w0" starttime="1.0" endtime="1.0" punc="true">.</w>
  <w nite:id="w1" starttime="5.0" endtime="5.4">Yeah</w>
  <w nite:id="w2" starttime="5.4" endtime="5.4" punc="true">.</w>
</nite:root>
"""
        words = load_words(_write(tmp_path, "w.xml", xml))
        sups = words_to_supervisions(
            words, meeting_id="ES2004a", channel=0, speaker="MEO015", utterance_gap=1.0
        )
        assert [s.text for s in sups] == ["Yeah."]
        assert all(any(ch.isalnum() for ch in s.text) for s in sups)


def _write_headsets(root: Path, mid: str, seconds: float, sr: int = 16000, drift: int = 0):
    audio_dir = root / "amicorpus" / mid / "audio"
    audio_dir.mkdir(parents=True)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    for ch in range(4):
        tone = (0.3 * np.sin(2 * np.pi * 300 * (ch + 1) * t)).astype(np.float32)
        if ch == 3 and drift:
            tone = tone[:-drift]
        sf.write(str(audio_dir / f"{mid}.Headset-{ch}.wav"), tone, sr, subtype="PCM_16")


class TestTranscode:
    def test_headset_paths(self, tmp_path):
        assert [p.name for p in headset_paths(tmp_path, "ES2004a")] == [
            f"ES2004a.Headset-{c}.wav" for c in range(4)
        ]

    def test_transcode_writes_four_channel_24k_flac_with_parity(self, tmp_path):
        _write_headsets(tmp_path, "ES2004a", seconds=3.0)
        rec = transcode_meeting(tmp_path, "ES2004a", tmp_path / "ami_flac")
        info = sf.info(str(tmp_path / "ami_flac" / "ES2004a.flac"))
        assert (info.channels, info.samplerate) == (4, 24000)
        assert abs(info.frames - round(3.0 * 16000 * 24000 / 16000)) <= 1
        loaded = load_ami_recordings(tmp_path, tmp_path / "ami_flac", ["ES2004a"])
        assert rec == loaded["ES2004a"]
        assert rec.audio_relpath == "ami_flac/ES2004a.flac"
        assert rec.num_channels == 4 and rec.sample_rate == 24000
        assert abs(rec.duration - 3.0) < 1e-3

    def test_transcode_rejects_headset_length_mismatch(self, tmp_path):
        _write_headsets(tmp_path, "ES2004a", seconds=3.0, drift=2 * 16000)
        with pytest.raises(ValueError, match="frame count"):
            transcode_meeting(tmp_path, "ES2004a", tmp_path / "ami_flac")

    def test_transcode_is_idempotent(self, tmp_path):
        _write_headsets(tmp_path, "ES2004a", seconds=1.0)
        a = transcode_meeting(tmp_path, "ES2004a", tmp_path / "ami_flac")
        target = tmp_path / "ami_flac" / "ES2004a.flac"
        mtime = target.stat().st_mtime_ns
        b = transcode_meeting(tmp_path, "ES2004a", tmp_path / "ami_flac")
        assert a == b
        assert target.stat().st_mtime_ns == mtime
