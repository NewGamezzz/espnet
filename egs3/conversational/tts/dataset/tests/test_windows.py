"""Tests for turn merging (AC3) and utterance-boundary windowing (AC4, AC5)."""

import gzip
import json
import random

import pytest

from egs3.conversational.tts.dataset.sssd import (
    Recording,
    Supervision,
    Turn,
    load_recordings,
    load_supervisions,
    merge_turns,
    occupied_intervals,
    session_speakers,
)
from egs3.conversational.tts.dataset.windows import (
    _gap_at,
    blocked_intervals,
    build_windows,
    first_eligible_boundary,
    from_json,
    is_eligible_boundary,
    select_window_spans,
    to_json,
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
        turns = merge_turns(
            [sup(0, 0.0, 1.0, "hello"), sup(0, 1.9, 3.0, "world")], MERGE_GAP
        )
        assert len(turns) == 1
        assert turns[0].text == "hello world"
        assert (turns[0].start, turns[0].end) == (0.0, 3.0)

    def test_gap_above_stays_separate(self):
        turns = merge_turns(
            [sup(0, 0.0, 1.0, "hello"), sup(0, 2.1, 3.0, "world")], MERGE_GAP
        )
        assert [t.text for t in turns] == ["hello", "world"]

    def test_gap_exactly_merge_gap_stays_separate(self):
        turns = merge_turns(
            [sup(0, 0.0, 1.0, "hello"), sup(0, 2.0, 3.0, "world")], MERGE_GAP
        )
        assert len(turns) == 2

    def test_overlapping_same_channel_spans_merge(self):
        turns = merge_turns(
            [sup(0, 0.0, 2.0, "hello"), sup(0, 1.5, 3.0, "world")], MERGE_GAP
        )
        assert len(turns) == 1
        assert turns[0].end == 3.0

    def test_cross_channel_never_merges(self):
        turns = merge_turns(
            [sup(0, 0.0, 1.0, "hello"), sup(1, 1.1, 2.0, "hi")], MERGE_GAP
        )
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


WINDOW_KW = dict(window_min=10.0, window_max=30.0, boundary_guard=0.0, tail_min=5.0)


def make_recording(duration, num_channels=2, rec_id="sess1"):
    return Recording(
        id=rec_id,
        audio_relpath=f"original/{rec_id}_mixed.flac",
        sample_rate=48000,
        num_channels=num_channels,
        duration=duration,
    )


def dialogue_turns(duration, turn_len=3.0, gap=1.0, num_channels=2):
    """Alternating clean dialogue: turn_len s of speech then gap s of silence."""
    turns = []
    t, i = 0.5, 0
    while t + turn_len + 0.5 < duration:
        turns.append(
            Turn(
                channel=i % num_channels,
                speaker=f"spk{i % num_channels}",
                text="hello there how are you",
                start=t,
                end=t + turn_len,
            )
        )
        t += turn_len + gap
        i += 1
    return turns


def turn(channel, start, end, text="hi there"):
    return Turn(
        channel=channel, speaker=f"spk{channel}", text=text, start=start, end=end
    )


class TestBoundaryEligibility:
    """The CoVoMix cut rule: eligible iff no turn (with guard) covers t."""

    def test_overlap_across_end_blocks_the_whole_region(self):
        # ch1 overlaps ch0's end: no boundary may fall anywhere in (5, 12),
        # in particular not at ch0's own end (t=10).
        blocked = blocked_intervals([turn(0, 5.0, 10.0), turn(1, 8.0, 12.0)], 0.0)
        assert blocked == [(5.0, 12.0)]
        for t in (6.0, 8.0, 10.0, 11.9):
            assert not is_eligible_boundary(blocked, t)
        assert is_eligible_boundary(blocked, 5.0)
        assert is_eligible_boundary(blocked, 12.0)

    def test_back_to_back_turns_yield_boundary_iff_guard_zero(self):
        turns = [turn(0, 0.0, 5.0), turn(1, 5.0, 9.0)]
        assert is_eligible_boundary(blocked_intervals(turns, 0.0), 5.0)
        assert first_eligible_boundary(blocked_intervals(turns, 0.0), 3.0) == 5.0
        assert not is_eligible_boundary(blocked_intervals(turns, 0.1), 5.0)

    def test_guard_rejects_activity_within_g_and_accepts_2g_gap(self):
        g = 0.5
        blocked = blocked_intervals([turn(0, 10.0, 20.0)], g)
        assert not is_eligible_boundary(blocked, 9.7)  # turn active inside (t-g, t+g)
        assert not is_eligible_boundary(blocked, 20.4)
        assert is_eligible_boundary(blocked, 9.5)
        assert is_eligible_boundary(blocked, 20.5)
        # A gap of exactly 2g centered on t is accepted...
        two_turns = [turn(0, 0.0, 10.0), turn(1, 11.0, 20.0)]
        assert is_eligible_boundary(blocked_intervals(two_turns, g), 10.5)
        # ... and rejected as soon as the guard exceeds half the gap.
        assert not is_eligible_boundary(blocked_intervals(two_turns, 0.6), 10.5)

    def test_merged_turn_is_never_split(self):
        # Raw supervisions have a 0.5 s gap below merge_gap; the merged turn
        # spans it, so no boundary may fall inside that internal gap.
        sups = [sup(0, 0.0, 2.0, "hello"), sup(0, 2.5, 5.0, "world")]
        turns = merge_turns(sups, MERGE_GAP)
        assert len(turns) == 1
        blocked = blocked_intervals(turns, 0.0)
        for t in (2.0, 2.2, 2.5):
            assert not is_eligible_boundary(blocked, t)

    def test_first_eligible_boundary_chains_through_overlaps(self):
        blocked = blocked_intervals(
            [turn(0, 1.0, 4.0), turn(1, 3.5, 8.0), turn(0, 8.0, 9.0)], 0.0
        )
        assert first_eligible_boundary(blocked, 2.0) == 8.0
        assert first_eligible_boundary(blocked, 0.5) == 0.5
        assert first_eligible_boundary(blocked, 8.0) == 8.0

    def test_gap_at(self):
        turns = [turn(0, 1.0, 3.0), turn(1, 3.0, 5.0), turn(0, 6.0, 7.0)]
        assert _gap_at(turns, 3.0) == pytest.approx(0.0)  # zero-gap exchange
        assert _gap_at(turns, 5.5) == pytest.approx(1.0)  # inside the 5-6 silence
        assert _gap_at(turns, 0.5) is None  # session edge: nothing before


class TestWindowIntegrity:
    """AC4: no turn strictly contains a boundary; durations in range; empty dropped."""

    def test_boundaries_and_durations(self):
        rec = make_recording(120.0)
        turns = dialogue_turns(120.0)
        rng = random.Random("0:window:sess1")
        records, stats = build_windows("sess1", rec, turns, rng=rng, **WINDOW_KW)
        assert records, "expected at least one window from a 2-minute session"
        for w in records:
            for t in turns:
                for edge in (w.t0, w.t1):
                    assert not (
                        t.start < edge < t.end
                    ), f"turn {t} strictly contains boundary {edge}"
            assert w.turns == tuple(
                t for t in turns if t.start >= w.t0 and t.end <= w.t1
            )
            assert len(w.turns) > 0
        durations = [w.duration for w in records]
        for d in durations[:-1]:
            assert WINDOW_KW["window_min"] <= d <= WINDOW_KW["window_max"] + 1e-6
        # The tail may be shorter than window_min but never below tail_min.
        assert WINDOW_KW["tail_min"] <= durations[-1] <= WINDOW_KW["window_max"] + 1e-6
        assert stats.n_windows == len(records)
        assert all(g >= 0 for g in stats.cut_gaps)

    def test_windows_tile_without_overlap(self):
        rec = make_recording(120.0)
        turns = dialogue_turns(120.0)
        records, _ = build_windows(
            "sess1", rec, turns, rng=random.Random("s"), **WINDOW_KW
        )
        for a, b in zip(records, records[1:]):
            assert a.t1 <= b.t0 + 1e-9

    def test_near_zero_gaps_are_now_windowable(self):
        # Continuous dialogue with 0.1 s gaps: unbreakable under the old
        # all-channel-silence rule, fully windowable at utterance boundaries.
        rec = make_recording(50.0)
        turns = dialogue_turns(50.0, turn_len=3.0, gap=0.1)
        records, stats = build_windows(
            "sess1", rec, turns, rng=random.Random("s"), **WINDOW_KW
        )
        assert records
        assert stats.dropped_span_sec == pytest.approx(0.0)

    def test_chained_overlap_region_dropped(self):
        # Turns overlap in an unbroken chain: no eligible boundary anywhere
        # inside, so the whole span is dropped rather than emitted oversize.
        rec = make_recording(50.0)
        turns = [
            turn(i % 2, start, start + 11.0)
            for i, start in enumerate([0.5, 10.0, 19.5, 29.0, 38.5])
        ]  # last ends at 49.5
        records, stats = build_windows(
            "sess1", rec, turns, rng=random.Random("s"), **WINDOW_KW
        )
        assert records == []
        assert stats.dropped_span_sec == pytest.approx(49.5)
        assert stats.dropped_tail_sec == pytest.approx(0.5)

    def test_session_shorter_than_window_min_is_tail(self):
        rec = make_recording(8.0)
        turns = dialogue_turns(8.0)
        records, _ = build_windows(
            "s1", rec, turns, rng=random.Random("s"), **WINDOW_KW
        )
        assert len(records) == 1
        assert (records[0].t0, records[0].t1) == (0.0, 8.0)

    def test_session_shorter_than_tail_min_dropped(self):
        rec = make_recording(3.0)
        turns = [turn(0, 0.5, 1.0)]
        records, stats = build_windows(
            "s1", rec, turns, rng=random.Random("s"), **WINDOW_KW
        )
        assert records == []
        assert stats.dropped_tail_sec == pytest.approx(3.0)

    def test_zero_speech_window_dropped(self):
        # 30 s of leading silence: the first window is speech-free and dropped.
        rec = make_recording(60.0)
        turns = [
            turn(0, 30.0, 33.0),
            turn(1, 34.0, 36.0),
            turn(0, 37.0, 40.0),
        ]
        records, stats = build_windows(
            "s1", rec, turns, rng=random.Random("s"), **WINDOW_KW
        )
        assert stats.dropped_empty_windows >= 1
        for w in records:
            assert len(w.turns) > 0


class TestWindowDeterminism:
    """AC5: same seed -> identical spans."""

    def test_same_seed_identical(self):
        rec = make_recording(300.0)
        turns = dialogue_turns(300.0)

        def run():
            return build_windows(
                "sess1", rec, turns, rng=random.Random("0:window:sess1"), **WINDOW_KW
            )

        records_a, stats_a = run()
        records_b, stats_b = run()
        assert records_a == records_b
        assert stats_a == stats_b

    def test_different_seed_differs(self):
        rec = make_recording(300.0)
        turns = dialogue_turns(300.0)
        spans_a, _ = build_windows(
            "sess1", rec, turns, rng=random.Random("seed-a"), **WINDOW_KW
        )
        spans_b, _ = build_windows(
            "sess1", rec, turns, rng=random.Random("seed-b"), **WINDOW_KW
        )
        assert [(w.t0, w.t1) for w in spans_a] != [(w.t0, w.t1) for w in spans_b]

    def test_json_roundtrip(self):
        rec = make_recording(120.0)
        turns = dialogue_turns(120.0)
        records, _ = build_windows(
            "sess1", rec, turns, rng=random.Random("s"), **WINDOW_KW
        )
        w = records[0]
        assert from_json(json.loads(json.dumps(to_json(w)))) == w


class TestSelectWindowSpans:
    def test_never_emits_oversize(self):
        # One long blocked region: the span up to its end is dropped, then
        # normal windows resume.
        blocked = [(5.0, 48.0)]
        kw = {k: v for k, v in WINDOW_KW.items() if k != "boundary_guard"}
        spans, stats = select_window_spans(blocked, 100.0, rng=random.Random("s"), **kw)
        for t0, t1 in spans:
            assert t1 - t0 <= WINDOW_KW["window_max"] + 1e-9
        assert stats.dropped_span_sec == pytest.approx(48.0)
        assert spans and spans[0][0] == 48.0
