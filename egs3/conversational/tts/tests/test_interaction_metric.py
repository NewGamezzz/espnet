"""``InteractionMetric`` (``src/metrics/interaction.py``) tests.

The dGSLM event derivation is pure interval arithmetic, so it is pinned
directly on constructed span layouts (no VAD, no audio). The metric round
trip runs with a keyed fake VAD and fabricated wavs, covering: pause vs gap
attribution, window-edge silence skipping, overlap merging, POOLED
per-minute rates (never a mean of per-window rates), W1 against the paired
ground truth read from the same meta (collapsing to ~0 when gen == gt,
i.e. the gt anchor mode), None semantics for empty distributions, backend
laziness, and offline hydra instantiation of the updated config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.interaction import (
    EVENT_TYPES,
    InteractionMetric,
    derive_events,
)
from egs3.conversational.tts.src.metrics.quality import SileroVADSegmenter

ALL_SUMMARY_KEYS = {
    f"{event}_{suffix}"
    for event in EVENT_TYPES
    for suffix in ("per_min", "sec_per_min", "dur_w1")
}


# --------------------------------------------------------------------------- #
# derive_events: pure interval arithmetic
# --------------------------------------------------------------------------- #
class TestDeriveEvents:
    def test_total_sec_only_shapes_the_skipped_trailing_silence(self):
        # External test sets with reference audio score gen and gt events
        # against ONE window duration (the generated one), while the gt wav
        # has its own length.  Every event must be independent of that
        # total: a trailing silence is a window-edge silence and is skipped,
        # and a total SHORTER than the last IPU must not clip or invent one.
        ipus = [[(0.0, 1.0), (3.0, 4.0)], [(1.5, 2.5)]]
        reference = derive_events(ipus, 4.0)
        for total in (2.0, 4.0, 4.5, 60.0):
            assert derive_events(ipus, total) == reference
        assert reference["gap"] == [pytest.approx(0.5), pytest.approx(0.5)]
        assert reference["pause"] == []

    def test_pause_same_speaker_on_both_sides(self):
        events = derive_events([[(0.0, 2.0), (3.0, 5.0)], []], total_sec=5.0)
        assert events["ipu"] == [2.0, 2.0]
        assert events["pause"] == [1.0]
        assert events["gap"] == []
        assert events["overlap"] == []

    def test_gap_floor_changes_hands(self):
        events = derive_events([[(0.0, 2.0)], [(3.0, 5.0)]], total_sec=5.0)
        assert events["pause"] == []
        assert events["gap"] == [1.0]

    def test_overlap_is_the_intersection(self):
        events = derive_events([[(0.0, 4.0)], [(2.0, 6.0)]], total_sec=6.0)
        assert events["overlap"] == [2.0]
        assert events["pause"] == [] and events["gap"] == []

    def test_window_edge_silences_are_skipped(self):
        # Leading silence has no before-speaker, trailing has no
        # after-speaker: neither is a pause or a gap.
        events = derive_events([[(1.0, 2.0)], []], total_sec=4.0)
        assert events["pause"] == [] and events["gap"] == []
        assert events["ipu"] == [1.0]

    def test_unsorted_and_touching_spans_are_merged_per_channel(self):
        # (2,3) then (0,2) arrive unsorted and touch: one 3 s IPU, and the
        # merge leaves no fake intra-channel silence.
        events = derive_events([[(2.0, 3.0), (0.0, 2.0)], []], total_sec=3.0)
        assert events["ipu"] == [3.0]
        assert events["pause"] == []

    def test_gap_after_overlap_attributes_by_ipu_ends(self):
        # ch0 speaks 0-3, ch1 overlaps 2-4, silence 4-5, then ch0 5-6:
        # the last IPU to end before the silence is ch1's -> floor moved
        # ch1 -> ch0 across the silence -> gap.
        events = derive_events([[(0.0, 3.0), (5.0, 6.0)], [(2.0, 4.0)]], total_sec=6.0)
        assert events["overlap"] == [1.0]
        assert events["gap"] == [1.0]
        assert events["pause"] == []


# --------------------------------------------------------------------------- #
# test-only fakes and fixture builders
# --------------------------------------------------------------------------- #
class KeyedFakeVADBackend:
    """Returns registered spans for the EXACT wav it's called with."""

    def __init__(self):
        self._table: dict[tuple, list] = {}

    @staticmethod
    def _key(wav) -> tuple:
        return tuple(np.round(np.asarray(wav, dtype=np.float64), 6).tolist())

    def register(self, wav: np.ndarray, spans) -> np.ndarray:
        arr = np.asarray(wav, dtype=np.float32)
        self._table[self._key(arr)] = list(spans)
        return arr

    def __call__(self, wav, sr):
        return self._table[self._key(wav)]


def _write_wav_exact(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def _unique_wav(seed: int, sr: int, duration_s: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration_s * sr)) * 0.1).astype(np.float32)


def _write_window(
    test_dir: Path,
    wid: str,
    sr: int,
    duration_sec: float,
    gen_wavs: list,
    gt_wavs: list,
) -> None:
    channels = []
    for ch, (gen, gt) in enumerate(zip(gen_wavs, gt_wavs)):
        gen_rel = f"wav/{wid}_ch{ch}.wav"
        gt_rel = f"gt/{wid}_ch{ch}.wav"
        _write_wav_exact(test_dir / gen_rel, gen, sr)
        _write_wav_exact(test_dir / gt_rel, gt, sr)
        channels.append(
            {
                "gen_wav": gen_rel,
                "prompt_wav": gen_rel,
                "gt_wav": gt_rel,
                "ref_text": "",
            }
        )
    meta = {
        "window_id": wid,
        "session_id": "sess",
        "mode": "generate",
        "sample_rate": sr,
        "num_channels": len(channels),
        "window_duration_sec": duration_sec,
        "rtf": None,
        "mix_wav": f"mix/{wid}.wav",
        "prompt": {"total_sec": 4.0, "total_frames": 375, "turns": []},
        "channels": channels,
        "turns": [],
    }
    (test_dir / "meta").mkdir(parents=True, exist_ok=True)
    (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_meta_scp(test_dir: Path, window_ids: list) -> None:
    lines = [f"{wid} meta/{wid}.json" for wid in window_ids]
    (test_dir / "meta.scp").write_text("".join(f"{line}\n" for line in lines))


# --------------------------------------------------------------------------- #
# backend laziness
# --------------------------------------------------------------------------- #
class TestBackendLaziness:
    def test_default_backend_is_lazy_silero(self):
        metric = InteractionMetric()
        assert isinstance(metric.vad_backend, SileroVADSegmenter)
        assert metric.vad_backend._get_speech_timestamps is None


# --------------------------------------------------------------------------- #
# full __call__ round trip
# --------------------------------------------------------------------------- #
class TestCallRoundTrip:
    SR = 16000

    def _one_window(self, tmp_path, gen_spans, gt_spans, duration_sec=10.0):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        vad = KeyedFakeVADBackend()
        gen_wavs, gt_wavs = [], []
        for ch in range(2):
            gen = _unique_wav(seed=10 + ch, sr=self.SR)
            gt = _unique_wav(seed=20 + ch, sr=self.SR)
            vad.register(gen, gen_spans[ch])
            vad.register(gt, gt_spans[ch])
            gen_wavs.append(gen)
            gt_wavs.append(gt)
        _write_window(test_dir, "sess_w00000", self.SR, duration_sec, gen_wavs, gt_wavs)
        _write_meta_scp(test_dir, ["sess_w00000"])
        return inference_dir, test_dir, vad

    def test_rates_w1_and_the_documented_keys(self, tmp_path):
        # gen: ch0 0-2, ch1 3-5 -> 2 IPUs, 1 gap (1 s), no pause/overlap.
        # gt: ch0 0-2 and 3-5 -> 2 IPUs, 1 pause; gap distribution empty.
        inference_dir, test_dir, vad = self._one_window(
            tmp_path,
            gen_spans=[[(0.0, 2.0)], [(3.0, 5.0)]],
            gt_spans=[[(0.0, 2.0), (3.0, 5.0)], []],
            duration_sec=10.0,
        )
        metric = InteractionMetric(vad_backend=vad)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert set(summary) == ALL_SUMMARY_KEYS
        sixth = 10.0 / 60.0  # window minutes
        assert summary["ipu_per_min"] == pytest.approx(2 / sixth)
        assert summary["ipu_sec_per_min"] == pytest.approx(4.0 / sixth)
        assert summary["gap_per_min"] == pytest.approx(1 / sixth)
        assert summary["pause_per_min"] == pytest.approx(0.0)
        assert summary["overlap_per_min"] == pytest.approx(0.0)
        # gen IPU durations [2,2] == gt [2,2] -> W1 0; gen has no pauses
        # and gt no gaps -> both W1 undefined (None, not 0).
        assert summary["ipu_dur_w1"] == pytest.approx(0.0)
        assert summary["pause_dur_w1"] is None
        assert summary["gap_dur_w1"] is None

        record = json.loads(
            (inference_dir / "valid" / "scoring" / "interaction" / "windows.jsonl")
            .read_text("utf-8")
            .splitlines()[0]
        )
        assert record["gen_events"]["gap"] == [1.0]
        assert record["gt_events"]["pause"] == [1.0]

    def test_gt_mode_collapse_w1_zero_when_gen_equals_gt(self, tmp_path):
        # In gt mode gen_wav IS the ground truth: identical spans on both
        # sides -> every defined W1 is exactly 0.
        spans = [[(0.0, 2.0)], [(3.0, 5.0)]]
        inference_dir, test_dir, vad = self._one_window(
            tmp_path, gen_spans=spans, gt_spans=spans
        )
        metric = InteractionMetric(vad_backend=vad)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)
        assert summary["ipu_dur_w1"] == pytest.approx(0.0)
        assert summary["gap_dur_w1"] == pytest.approx(0.0)

    def test_rates_pool_over_total_minutes_not_mean_of_window_rates(self, tmp_path):
        # Window A: 60 s with 3 gen gaps (alternating floor); window B:
        # 30 s with none. Pooled: 3 gaps / 1.5 min = 2.0/min. A mean of
        # per-window rates would give (3/1 + 0/0.5)/2 = 1.5 -- the
        # distinguishing value.
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        vad = KeyedFakeVADBackend()

        layouts = {
            "sess_w00000": (
                60.0,
                # ch0 [0,2],[6,8]; ch1 [3,5],[9,11]: silences 2-3, 5-6, 8-9
                # all change floor -> 3 gaps; 11-60 is edge, skipped.
                [[(0.0, 2.0), (6.0, 8.0)], [(3.0, 5.0), (9.0, 11.0)]],
            ),
            "sess_w00001": (30.0, [[(0.0, 2.0)], []]),
        }
        seed = 0
        for wid, (dur, gen_spans) in layouts.items():
            gen_wavs, gt_wavs = [], []
            for ch in range(2):
                gen = _unique_wav(seed=100 + seed, sr=sr)
                gt = _unique_wav(seed=200 + seed, sr=sr)
                seed += 1
                vad.register(gen, gen_spans[ch])
                vad.register(gt, gen_spans[ch])
                gen_wavs.append(gen)
                gt_wavs.append(gt)
            _write_window(test_dir, wid, sr, dur, gen_wavs, gt_wavs)
        _write_meta_scp(test_dir, list(layouts))

        metric = InteractionMetric(vad_backend=vad)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert summary["gap_per_min"] == pytest.approx(3 / 1.5)


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
        assert metric.vad_backend._get_speech_timestamps is None
