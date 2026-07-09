"""Tests for turn merging (AC3) and silence-aligned windowing (AC4, AC5)."""

import gzip
import json

import pytest

from egs3.conversational.tts.dataset.sssd import (
    Supervision,
    load_recordings,
    load_supervisions,
    merge_turns,
    occupied_intervals,
    session_speakers,
)

MERGE_GAP = 1.0


def sup(channel, start, end, text, speaker=None, rec="rec1", idx=[0]):
    idx[0] += 1
    return Supervision(
        id=f"utt{idx[0]:04d}",
        recording_id=rec,
        channel=channel,
        start=start,
        duration=end - start,
        text=text,
        speaker=speaker or f"spk{channel}",
    )


class TestMergeTurns:
    """AC3: merge below merge_gap, stay separate at/above, single-space join."""

    def test_gap_below_merges(self):
        turns = merge_turns([sup(0, 0.0, 1.0, "hello"), sup(0, 1.9, 3.0, "world")], MERGE_GAP)
        assert len(turns) == 1
        assert turns[0].text == "hello world"
        assert (turns[0].start, turns[0].end) == (0.0, 3.0)

    def test_gap_above_stays_separate(self):
        turns = merge_turns([sup(0, 0.0, 1.0, "hello"), sup(0, 2.1, 3.0, "world")], MERGE_GAP)
        assert [t.text for t in turns] == ["hello", "world"]

    def test_gap_exactly_merge_gap_stays_separate(self):
        turns = merge_turns([sup(0, 0.0, 1.0, "hello"), sup(0, 2.0, 3.0, "world")], MERGE_GAP)
        assert len(turns) == 2

    def test_overlapping_same_channel_spans_merge(self):
        turns = merge_turns([sup(0, 0.0, 2.0, "hello"), sup(0, 1.5, 3.0, "world")], MERGE_GAP)
        assert len(turns) == 1
        assert turns[0].end == 3.0

    def test_cross_channel_never_merges(self):
        turns = merge_turns([sup(0, 0.0, 1.0, "hello"), sup(1, 1.1, 2.0, "hi")], MERGE_GAP)
        assert len(turns) == 2
        assert [t.channel for t in turns] == [0, 1]

    def test_backchannel_does_not_split_turn(self):
        turns = merge_turns(
            [
                sup(0, 0.0, 2.0, "so i was thinking"),
                sup(1, 2.1, 2.4, "mm-hmm"),
                sup(0, 2.5, 4.0, "that we could go"),
            ],
            MERGE_GAP,
        )
        ch0 = [t for t in turns if t.channel == 0]
        assert len(ch0) == 1
        assert ch0[0].text == "so i was thinking that we could go"

    def test_global_order_sorted_by_start_then_channel(self):
        turns = merge_turns(
            [sup(1, 0.0, 1.0, "b"), sup(0, 0.0, 1.0, "a"), sup(0, 5.0, 6.0, "c")],
            MERGE_GAP,
        )
        assert [(t.start, t.channel) for t in turns] == [(0.0, 0), (0.0, 1), (5.0, 0)]

    def test_three_channels(self):
        turns = merge_turns(
            [sup(2, 0.0, 1.0, "x"), sup(0, 1.2, 2.0, "y"), sup(1, 2.2, 3.0, "z")],
            MERGE_GAP,
        )
        assert [t.channel for t in turns] == [2, 0, 1]


class TestOccupiedIntervals:
    def test_union_across_channels(self):
        turns = merge_turns(
            [sup(0, 0.0, 2.0, "a"), sup(1, 1.5, 3.0, "b"), sup(0, 10.0, 11.0, "c")],
            MERGE_GAP,
        )
        assert occupied_intervals(turns) == [(0.0, 3.0), (10.0, 11.0)]


class TestManifestParsing:
    def _write_manifests(self, tmp_path):
        rec_path = tmp_path / "recordings.jsonl.gz"
        sup_path = tmp_path / "supervisions.jsonl.gz"
        recording = {
            "id": "sess1",
            "sources": [
                {
                    "type": "file",
                    "channels": [0, 1],
                    "source": "/scratch/bbjs/cornell2/SSSD/original/sess1_mixed.flac",
                }
            ],
            "sampling_rate": 48000,
            "num_samples": 480000,
            "duration": 10.0,
            "channel_ids": [0, 1],
        }
        sups = [
            {
                "id": "sess1_ch0_utt0000",
                "recording_id": "sess1",
                "start": 0.5,
                "duration": 1.0,
                "channel": 0,
                "text": "can you hear me?",
                "speaker": "a" * 64,
                "alignment": {"word": [["can", 0.5, 0.3, None]]},
            },
            {
                "id": "sess1_ch1_utt0000",
                "recording_id": "sess1",
                "start": 2.0,
                "duration": 99.0,  # exceeds recording duration -> clamped
                "channel": 1,
                "text": "yes",
                "speaker": "b" * 64,
            },
        ]
        with gzip.open(rec_path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(recording) + "\n")
        with gzip.open(sup_path, "wt", encoding="utf-8") as f:
            for s in sups:
                f.write(json.dumps(s) + "\n")
        return rec_path, sup_path

    def test_roundtrip_with_remap_and_clamp(self, tmp_path):
        rec_path, sup_path = self._write_manifests(tmp_path)
        recordings = load_recordings(rec_path)
        assert recordings["sess1"].audio_relpath == "original/sess1_mixed.flac"
        assert recordings["sess1"].num_channels == 2
        sups = load_supervisions(sup_path, recordings)["sess1"]
        assert [s.channel for s in sups] == [0, 1]
        assert sups[1].end == pytest.approx(10.0)  # clamped to duration
        assert session_speakers(sups) == {"a" * 64, "b" * 64}

    def test_multi_source_recording_raises(self, tmp_path):
        rec_path = tmp_path / "recordings.jsonl.gz"
        recording = {
            "id": "bad",
            "sources": [
                {"type": "file", "channels": [0], "source": "/x/a.flac"},
                {"type": "file", "channels": [1], "source": "/x/b.flac"},
            ],
            "sampling_rate": 48000,
            "duration": 5.0,
            "channel_ids": [0, 1],
        }
        with gzip.open(rec_path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(recording) + "\n")
        with pytest.raises(ValueError, match="sources"):
            load_recordings(rec_path)
