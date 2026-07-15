"""``InteractionMetric`` (``src/metrics/interaction.py``) tests.

Two layers, mirroring ``tests/test_speaker_metric.py``'s structure:

1. Pure interval-algebra tests directly on hand-constructed ``channel_ipus``
   (``List[List[(start, end)]]``) -- no audio, no VAD -- covering the five
   scenarios the task calls out by name: a clean turn exchange (gap), a
   same-speaker pause, a partial overlap, a backchannel IPU inside another
   channel's long turn, and a leading/trailing silence that must count as
   neither gap nor pause. Every expected count/duration below is
   hand-computed in the test's own comment, not just asserted against the
   implementation.
2. A handful of full ``__call__`` round trips (amplitude-envelope wavs +
   an energy-threshold fake VAD) proving the wiring: meta-relative path
   resolution, the JSONL/summary schema, backend laziness, laughter gating,
   and offline ``conf/metrics.yaml`` instantiation.

CPU-only, no network throughout (the real default VAD is silero, gated
lazily behind the first call; never exercised here).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.interaction import (
    InteractionMetric,
    _active_count_segments,
    _classify_silence,
    _compute_event_battery,
    _count_backchannels,
    _merge_predicate_intervals,
    _overlap_with_other_channels,
    _rate_stats,
    _w1,
)
from egs3.conversational.tts.src.metrics.segments import VAD, SileroVADBackend
from egs3.conversational.tts.tests.conftest import _FrameEnergyVAD


def _write_wav_exact(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def _block(duration_s: float, amplitude: float, sr: int) -> np.ndarray:
    return np.full(int(round(duration_s * sr)), amplitude, dtype=np.float32)


def _silence(duration_s: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(duration_s * sr)), dtype=np.float32)


# --------------------------------------------------------------------------- #
# _active_count_segments / _merge_predicate_intervals: the sweep-line core
# --------------------------------------------------------------------------- #
class TestActiveCountSegments:
    def test_two_channel_partial_overlap_produces_expected_counts(self):
        # ch0: [0,2), ch1: [1,3) -> counts 1 on [0,1), 2 on [1,2), 1 on [2,3)
        segments = _active_count_segments([[(0.0, 2.0)], [(1.0, 3.0)]], 3.0)
        assert segments == [(0.0, 1.0, 1), (1.0, 2.0, 2), (2.0, 3.0, 1)]

    def test_gap_between_channels_produces_a_zero_count_segment(self):
        segments = _active_count_segments([[(0.0, 1.0)], [(1.5, 2.5)]], 2.5)
        assert segments == [(0.0, 1.0, 1), (1.0, 1.5, 0), (1.5, 2.5, 1)]


class TestMergePredicateIntervals:
    def test_overlap_predicate_merges_only_count_ge_2(self):
        segments = [(0.0, 1.0, 1), (1.0, 2.0, 2), (2.0, 3.0, 1)]
        assert _merge_predicate_intervals(segments, lambda c: c >= 2) == [(1.0, 2.0)]

    def test_silence_predicate_merges_only_count_eq_0(self):
        segments = [(0.0, 1.0, 1), (1.0, 1.5, 0), (1.5, 2.5, 1)]
        assert _merge_predicate_intervals(segments, lambda c: c == 0) == [(1.0, 1.5)]

    def test_adjacent_accepted_segments_merge_into_one_interval(self):
        segments = [(0.0, 1.0, 0), (1.0, 2.0, 0), (2.0, 3.0, 1)]
        assert _merge_predicate_intervals(segments, lambda c: c == 0) == [(0.0, 2.0)]


# --------------------------------------------------------------------------- #
# _classify_silence: gap vs. pause vs. neither, including the tie case
# --------------------------------------------------------------------------- #
class TestClassifySilence:
    def test_different_preceding_and_following_channel_is_a_gap(self):
        channel_ipus = [[(0.0, 1.0)], [(1.5, 2.5)]]
        assert _classify_silence(channel_ipus, 1.0, 1.5, 2.5) == "gap"

    def test_same_single_channel_before_and_after_is_a_pause(self):
        channel_ipus = [[(0.0, 1.0), (1.5, 2.5)], []]
        assert _classify_silence(channel_ipus, 1.0, 1.5, 2.5) == "pause"

    def test_leading_silence_is_neither(self):
        channel_ipus = [[(1.0, 2.0)]]
        assert _classify_silence(channel_ipus, 0.0, 1.0, 3.0) is None

    def test_trailing_silence_is_neither(self):
        channel_ipus = [[(0.0, 1.0)]]
        assert _classify_silence(channel_ipus, 1.0, 3.0, 3.0) is None

    def test_whole_window_silence_is_neither(self):
        assert _classify_silence([[], []], 0.0, 2.0, 2.0) is None

    def test_tied_preceding_channels_intersecting_following_is_a_pause(self):
        # ch0 and ch1 BOTH end simultaneously at t=1.0 (a genuine tie in the
        # "preceding channel" set); only ch0 resumes at t=1.5. Per the
        # module docstring's set-based generalization: preceding={0,1},
        # following={0}, intersection={0} is non-empty -> pause (speaker 0
        # continuing), even though the tie means "the" preceding channel is
        # not literally unique.
        channel_ipus = [[(0.0, 1.0), (1.5, 2.5)], [(0.0, 1.0)]]
        assert _classify_silence(channel_ipus, 1.0, 1.5, 2.5) == "pause"

    def test_tied_preceding_channels_disjoint_from_following_channel_is_a_gap(self):
        # ch0 and ch1 both end at t=1.0; a THIRD channel (ch2) resumes at
        # t=1.5 -> preceding={0,1}, following={2}, disjoint -> gap.
        channel_ipus = [[(0.0, 1.0)], [(0.0, 1.0)], [(1.5, 2.5)]]
        assert _classify_silence(channel_ipus, 1.0, 1.5, 2.5) == "gap"


# --------------------------------------------------------------------------- #
# _compute_event_battery: the five named scenarios, hand-computed end to end
# --------------------------------------------------------------------------- #
class TestComputeEventBattery:
    def test_clean_turn_exchange_is_one_gap(self):
        # ch0 speaks [0,1), silence [1,1.5), ch1 speaks [1.5,2.5) -> a 0.5s
        # gap (different speakers on either side); region exactly covered at
        # both ends so no leading/trailing silence.
        channel_ipus = [[(0.0, 1.0)], [(1.5, 2.5)]]
        battery = _compute_event_battery(channel_ipus, 2.5)
        assert battery.ipu_durations == pytest.approx([1.0, 1.0])
        assert battery.gap_durations == pytest.approx([0.5])
        assert battery.pause_durations == []
        assert battery.overlap_durations == []

    def test_same_speaker_pause(self):
        # ch0 speaks [0,1), silence [1,1.5), ch0 speaks again [1.5,2.5) ->
        # a 0.5s pause (same speaker on both sides).
        channel_ipus = [[(0.0, 1.0), (1.5, 2.5)], []]
        battery = _compute_event_battery(channel_ipus, 2.5)
        assert battery.ipu_durations == pytest.approx([1.0, 1.0])
        assert battery.pause_durations == pytest.approx([0.5])
        assert battery.gap_durations == []
        assert battery.overlap_durations == []

    def test_partial_overlap(self):
        # ch0 [0,2), ch1 [1,3) -> overlap exactly [1,2), 1.0s; no silence at
        # all (active from 0 straight through to region end).
        channel_ipus = [[(0.0, 2.0)], [(1.0, 3.0)]]
        battery = _compute_event_battery(channel_ipus, 3.0)
        assert battery.overlap_durations == pytest.approx([1.0])
        assert battery.gap_durations == []
        assert battery.pause_durations == []
        assert battery.ipu_durations == pytest.approx([2.0, 2.0])

    def test_leading_and_trailing_silence_count_as_neither(self):
        # ch0 speaks only [1,2) inside a [0,3) region -> 1s leading silence
        # and 1s trailing silence, BOTH excluded from gap_durations and
        # pause_durations (not just one of the two).
        channel_ipus = [[(1.0, 2.0)]]
        battery = _compute_event_battery(channel_ipus, 3.0)
        assert battery.gap_durations == []
        assert battery.pause_durations == []
        assert battery.ipu_durations == pytest.approx([1.0])


# --------------------------------------------------------------------------- #
# backchannel proxy: short IPU mostly inside another channel's long turn
# --------------------------------------------------------------------------- #
class TestOverlapWithOtherChannels:
    def test_fully_contained_ipu_overlaps_its_own_duration(self):
        channel_ipus = [[(0.0, 5.0)], [(2.0, 2.3)]]
        overlap = _overlap_with_other_channels((2.0, 2.3), 1, channel_ipus)
        assert overlap == pytest.approx(0.3)

    def test_own_channel_is_excluded(self):
        channel_ipus = [[(0.0, 1.0), (2.0, 2.3)]]
        overlap = _overlap_with_other_channels((2.0, 2.3), 0, channel_ipus)
        assert overlap == pytest.approx(0.0)


class TestCountBackchannels:
    def test_short_ipu_fully_inside_another_channels_long_turn_counts(self):
        # ch0 speaks [0,5) (a long turn); ch1 has a 0.3s IPU [2,2.3)
        # entirely inside it -> overlap_frac = 0.3/0.3 = 1.0 >= 0.5, and
        # 0.3s < the default 1.0s backchannel_max_sec -> one backchannel.
        # ch0's own 5s IPU is far too long to qualify.
        channel_ipus = [[(0.0, 5.0)], [(2.0, 2.3)]]
        assert _count_backchannels(channel_ipus, 1.0, 0.5) == 1

    def test_ipu_at_or_above_max_sec_does_not_count(self):
        channel_ipus = [[(0.0, 5.0)], [(2.0, 3.0)]]  # 1.0s IPU, at the floor
        assert _count_backchannels(channel_ipus, 1.0, 0.5) == 0

    def test_overlap_fraction_exactly_at_threshold_counts(self):
        # ch1's 1.0s IPU overlaps ch0 by exactly 0.5s -> frac == 0.5, which
        # the ">= 0.5" rule counts.
        channel_ipus = [[(0.0, 2.5)], [(2.0, 3.0)]]
        assert _count_backchannels(channel_ipus, 1.5, 0.5) == 1

    def test_overlap_fraction_just_under_threshold_does_not_count(self):
        channel_ipus = [[(0.0, 2.49)], [(2.0, 3.0)]]  # overlap 0.49/1.0
        assert _count_backchannels(channel_ipus, 1.5, 0.5) == 0

    def test_no_overlap_at_all_does_not_count(self):
        channel_ipus = [[(0.0, 1.0)], [(2.0, 2.3)]]
        assert _count_backchannels(channel_ipus, 1.0, 0.5) == 0


# --------------------------------------------------------------------------- #
# per-minute rate stats
# --------------------------------------------------------------------------- #
class TestRateStats:
    def test_hand_computed_rate_and_duration_rate(self):
        rate, dur_rate = _rate_stats([1.0, 2.0], duration_minutes=2.0)
        assert rate == pytest.approx(1.0)  # 2 events / 2 min
        assert dur_rate == pytest.approx(1.5)  # 3.0s summed / 2 min

    def test_zero_duration_minutes_returns_none_none(self):
        assert _rate_stats([1.0], duration_minutes=0.0) == (None, None)

    def test_empty_durations_is_zero_rate_not_none(self):
        rate, dur_rate = _rate_stats([], duration_minutes=1.0)
        assert (rate, dur_rate) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# W1: hand-computed against scipy's own definition
# --------------------------------------------------------------------------- #
class TestW1:
    def test_single_element_distributions_is_the_absolute_difference(self):
        assert _w1([1.0], [3.0]) == pytest.approx(2.0)

    def test_matches_scipy_reference_on_a_multi_point_case(self):
        from scipy.stats import wasserstein_distance

        gen = [1.0, 3.0, 3.5]
        gt = [2.0, 2.0, 4.0]
        assert _w1(gen, gt) == pytest.approx(wasserstein_distance(gen, gt))

    def test_empty_generated_side_skips_not_zero(self):
        assert _w1([], [1.0]) is None

    def test_empty_ground_truth_side_skips_not_zero(self):
        assert _w1([1.0], []) is None

    def test_both_empty_skips(self):
        assert _w1([], []) is None


# --------------------------------------------------------------------------- #
# backend laziness
# --------------------------------------------------------------------------- #
class TestBackendLaziness:
    def test_metric_construction_with_all_real_defaults_does_not_touch_network(self):
        metric = InteractionMetric()
        assert isinstance(metric.vad, VAD)
        assert isinstance(metric.vad.backend, SileroVADBackend)
        assert metric.vad.backend._model is None
        assert metric.laughter_detector is None

    def test_construction_never_touches_torch_hub(self, monkeypatch):
        import torch.hub

        def guard(*args, **kwargs):
            raise AssertionError("torch.hub.load called during construction")

        monkeypatch.setattr(torch.hub, "load", guard)
        metric = InteractionMetric()
        assert metric.vad.backend._model is None


# --------------------------------------------------------------------------- #
# full __call__ round trip: JSONL/summary schema, meta-relative paths,
# laughter gating on/off.
# --------------------------------------------------------------------------- #
class TestCallRoundTrip:
    SR = 16000

    def _write_window(
        self,
        test_dir: Path,
        wid: str,
        ch0_gen: np.ndarray,
        ch1_gen: np.ndarray,
        boundary: float = 5.0,
    ) -> float:
        sr = self.SR
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch0.wav", ch0_gen, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch1.wav", ch1_gen, sr)
        # gt_wav: identical to gen here (round trip only needs a valid,
        # same-length file; the event-battery math is exercised separately
        # by the pure-function tests above).
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch0.wav", ch0_gen, sr)
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch1.wav", ch1_gen, sr)
        region_duration = len(ch0_gen) / sr
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + region_duration,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"wav/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"wav/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "",
                },
            ],
            "turns": [],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")
        (test_dir / "meta.scp").write_text(
            f"{wid} meta/{wid}.json\n", encoding="utf-8"
        )
        return region_duration

    def test_writes_jsonl_and_summary_with_the_documented_keys(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        ch0 = np.concatenate(
            [_block(1.0, 0.3, sr), _silence(0.5, sr), _silence(1.0, sr)]
        )
        ch1 = np.concatenate(
            [_silence(1.5, sr), _block(1.0, 0.3, sr), _silence(0.0, sr)]
        )
        self._write_window(test_dir, "sess_w00000", ch0, ch1)

        metric = InteractionMetric(vad=_FrameEnergyVAD(), min_silence=0.2)
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        assert set(summary) == {
            "ipu_per_min",
            "pause_per_min",
            "gap_per_min",
            "overlap_per_min",
            "ipu_dur_per_min",
            "pause_dur_per_min",
            "gap_dur_per_min",
            "overlap_dur_per_min",
            "w1_ipu",
            "w1_pause",
            "w1_gap",
            "w1_overlap",
            "backchannel_per_min",
        }
        # w1_* keys legitimately go undefined (None, not a fabricated 0.0)
        # when an event type has zero events on either side in the run's
        # only window -- see _common.py's summary_value.
        assert all(isinstance(v, float) or v is None for v in summary.values())

        scoring_dir = inference_dir / "valid" / "scoring" / "interaction"
        lines = (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["window_id"] == "sess_w00000"
        assert "laughter_per_min" not in record

        summary_text = (scoring_dir / "summary.json").read_text("utf-8")
        on_disk_summary = json.loads(summary_text)
        assert on_disk_summary == summary
        # Any undefined key above reaches disk as JSON null and would
        # render as "-" in local/eval_report.py; pin that for whichever
        # key(s) are actually undefined in this fixture.
        undefined_keys = [k for k, v in summary.items() if v is None]
        assert undefined_keys, "fixture no longer exercises an undefined key"
        for key in undefined_keys:
            assert f'"{key}": null' in summary_text

    def test_meta_relative_paths_resolve_and_gap_is_detected_end_to_end(
        self, tmp_path
    ):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        # ch0 speaks first second, ch1 speaks the next -> one gap of ~0.5s
        # once build_ipus (min_silence=0.2) keeps the silence undivided.
        ch0 = np.concatenate([_block(1.0, 0.3, sr), _silence(1.5, sr)])
        ch1 = np.concatenate([_silence(1.5, sr), _block(1.0, 0.3, sr)])
        region_duration = self._write_window(test_dir, "sess_w00000", ch0, ch1)
        duration_minutes = region_duration / 60.0

        metric = InteractionMetric(vad=_FrameEnergyVAD(), min_silence=0.2)
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        scoring_dir = inference_dir / "valid" / "scoring" / "interaction"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert record["generated"]["gap_durations"] == pytest.approx([0.5], abs=1e-2)
        assert record["generated"]["pause_durations"] == []
        expected_gap_per_min = 1.0 / duration_minutes
        assert record["gap_per_min"] == pytest.approx(expected_gap_per_min, rel=1e-2)
        assert summary["gap_per_min"] == pytest.approx(expected_gap_per_min, rel=1e-2)

    def test_laughter_detector_disabled_by_default_no_summary_keys(self, tmp_path):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        sr = self.SR
        ch0 = _silence(1.0, sr)
        ch1 = _silence(1.0, sr)
        self._write_window(test_dir, "sess_w00000", ch0, ch1)

        metric = InteractionMetric(vad=_FrameEnergyVAD())
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)
        assert "laughter_per_min" not in summary
        assert "laughter_mean_dur" not in summary

    def test_injected_laughter_detector_produces_hand_computed_rate_and_mean_dur(
        self, tmp_path
    ):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        sr = self.SR
        ch0 = _silence(1.0, sr)
        ch1 = _silence(1.0, sr)
        region_duration = self._write_window(test_dir, "sess_w00000", ch0, ch1)
        duration_minutes = region_duration / 60.0

        # fake detector: ch0 -> one 0.2s laughter event, ch1 -> one 0.4s
        # event -> 2 events total, mean duration (0.2+0.4)/2 = 0.3s.
        calls = {"n": 0}

        def fake_detector(wav, sr):
            calls["n"] += 1
            return [(0.0, 0.2)] if calls["n"] == 1 else [(0.0, 0.4)]

        metric = InteractionMetric(
            vad=_FrameEnergyVAD(), laughter_detector=fake_detector
        )
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert calls["n"] == 2  # once per generated channel
        expected_rate = 2 / duration_minutes
        assert summary["laughter_per_min"] == pytest.approx(expected_rate)
        assert summary["laughter_mean_dur"] == pytest.approx(0.3)

        scoring_dir = inference_dir / "valid" / "scoring" / "interaction"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert record["laughter_count"] == 2
        assert record["laughter_mean_dur"] == pytest.approx(0.3)

    def test_w1_wiring_uses_the_paired_gt_wav_not_the_gen_wav_twice(self, tmp_path):
        """Every other round-trip test in this class writes an IDENTICAL
        ``gt_wav``/``gen_wav`` pair, so a bug that fed ``gen_battery``'s own
        durations into BOTH sides of ``_w1`` (e.g. a copy-paste that reused
        ``gen_ipus`` where ``gt_ipus`` belonged) would pass every one of
        them silently -- W1 is a symmetric distance, so it cannot be caught
        by value alone either. This test gives channel 0 a genuinely
        DIFFERENT generated vs. ground-truth speech pattern (gen: two 1.0s
        IPUs split by an internal pause; gt: one continuous 2.6s IPU
        spanning the same extent) and asserts, independently: (a) the
        per-window JSONL's raw ``generated``/``ground_truth`` diagnostic
        duration lists each match their OWN source file (proving the two
        conditions were segmented from two different wavs, not one reused
        twice), and (b) ``w1_ipu`` equals ``scipy.stats.
        wasserstein_distance`` applied to exactly those two lists (proving
        the summary key is wired from the same two lists, not recomputed
        from something else)."""
        from scipy.stats import wasserstein_distance

        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        wid = "sess_w00000"

        # gen: speech [0,1), pause [1,1.6) (0.6s > 0.2 min_silence -> stays
        # split), speech [1.6,2.6), silence to region end.
        gen_ch0 = np.concatenate(
            [
                _block(1.0, 0.3, sr),
                _silence(0.6, sr),
                _block(1.0, 0.3, sr),
                _silence(0.4, sr),
            ]
        )
        # gt: one continuous 2.6s IPU over the same extent, same trailing
        # silence -> a materially different IPU-duration distribution.
        gt_ch0 = np.concatenate([_block(2.6, 0.3, sr), _silence(0.4, sr)])
        ch1 = _silence(3.0, sr)  # inert second channel

        _write_wav_exact(test_dir / "wav" / f"{wid}_ch0.wav", gen_ch0, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch1.wav", ch1, sr)
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch0.wav", gt_ch0, sr)
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch1.wav", ch1, sr)
        region_duration = len(gen_ch0) / sr
        boundary = 5.0
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + region_duration,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"wav/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"wav/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "",
                },
            ],
            "turns": [],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")
        (test_dir / "meta.scp").write_text(
            f"{wid} meta/{wid}.json\n", encoding="utf-8"
        )

        metric = InteractionMetric(vad=_FrameEnergyVAD(), min_silence=0.2)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        scoring_dir = inference_dir / "valid" / "scoring" / "interaction"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )

        gen_ipu_durs = record["generated"]["ipu_durations"]
        gt_ipu_durs = record["ground_truth"]["ipu_durations"]
        assert sorted(gen_ipu_durs) == pytest.approx([1.0, 1.0], abs=1e-2)
        assert gt_ipu_durs == pytest.approx([2.6], abs=1e-2)
        assert gen_ipu_durs != pytest.approx(gt_ipu_durs)  # genuinely distinct

        expected_w1 = wasserstein_distance(gen_ipu_durs, gt_ipu_durs)
        assert expected_w1 > 0.0  # sanity: the hand-built patterns do differ
        assert record["w1_ipu"] == pytest.approx(expected_w1, abs=1e-2)
        assert summary["w1_ipu"] == pytest.approx(expected_w1, abs=1e-2)
        assert "ipu" not in record["w1_skipped"]

        # gen has an internal pause the gt does not -> also exercises the
        # w1_pause skip+count convention for the gt side (gt has zero
        # pauses -> skipped, not fabricated as 0).
        assert record["generated"]["pause_durations"] == pytest.approx(
            [0.6], abs=1e-2
        )
        assert record["ground_truth"]["pause_durations"] == []
        assert record["w1_pause"] is None
        assert "pause" in record["w1_skipped"]


# --------------------------------------------------------------------------- #
# conf/metrics.yaml wiring: offline instantiation
# --------------------------------------------------------------------------- #
class TestMetricsConfigInstantiatesOffline:
    def test_interaction_metric_entry_instantiates_without_network(self, monkeypatch):
        from hydra.utils import instantiate

        from egs3.conversational.tts import run
        from espnet3.utils.config_utils import load_and_merge_config

        recipe_dir = Path(run.__file__).resolve().parent
        monkeypatch.chdir(recipe_dir)
        metrics_config = load_and_merge_config(
            Path("conf/metrics.yaml"),
            config_name=run.DEFAULT_METRICS_CONFIG,
            resolve=False,
        )

        entries = [
            entry
            for entry in metrics_config.metrics
            if entry.metric._target_.endswith("InteractionMetric")
        ]
        assert len(entries) == 1
        metric = instantiate(entries[0].metric)
        assert isinstance(metric, InteractionMetric)
        assert metric.vad.backend._model is None
        assert metric.laughter_detector is None
