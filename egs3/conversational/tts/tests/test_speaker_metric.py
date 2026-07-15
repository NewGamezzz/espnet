"""``SpeakerDynamicsMetric`` (``src/metrics/speaker.py``) tests.

Fake-embedder / fake-VAD, CPU-only, no network: covers the pure aggregation
math (cosine, least-squares slope, percentile), the ground-truth-turn ->
generated-region solo-interval clipping used by the bleed-dB leg, per-channel
feature extraction (concatenation for SIM-o, the ``embed_min_sec`` floor and
its documented asymmetry between the concatenated embedding and per-IPU
embeddings), bleed-dB arithmetic on synthetic two-channel audio with a known
injected leak, backend laziness, and a full ``__call__`` round trip against a
fabricated ``inference_dir`` mirroring ``tests/test_asr_metric.py``'s fixture
pattern.

Real WavLM-SV (``transformers``) is only exercised by the asset-gated smoke
test at the bottom.

The fake embedder used throughout (``KeyedFakeEmbedder``) keys on the EXACT
sample content of the array it's called with (not a summary statistic like
its mean), so a concatenation of several snippets can never collide with one
of its own constituents (its content differs by construction -- it's longer
and contains all of them). Test wavs are written with ``subtype="FLOAT"`` so
the float32 samples round-trip through disk bit-exactly (the default 16-bit
PCM subtype would quantize them and break the exact-content key).
"""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.speaker import (
    ChannelFeatures,
    SpeakerDynamicsMetric,
    WavLMSVEmbedder,
    _clip_turns_to_region,
    _confusion_pairs,
    _consistency,
    _cosine,
    _drift,
    _least_squares_slope,
    _percentile,
    _region_sumsq,
    _sim_o,
)

try:
    import transformers  # noqa: F401

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# --------------------------------------------------------------------------- #
# test-only fakes
# --------------------------------------------------------------------------- #
class _ListVAD:
    def __init__(self, segments):
        self._segments = segments

    def __call__(self, wav, sr):
        return self._segments


class _EnergyVAD:
    """Fake VAD that returns the whole snippet as one IPU when it has any
    energy, else none -- lets a fixture differentiate a "speaking" channel
    from a silent one without hand-listing per-channel VAD segments."""

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold

    def __call__(self, wav, sr):
        if np.max(np.abs(wav)) > self.threshold:
            return [(0.0, len(wav) / sr)]
        return []


class _FrameEnergyVAD:
    """Frame-wise energy-threshold VAD (mirrors ``test_segments.py``'s
    ``EnergyVADBackend``): unlike ``_EnergyVAD`` this actually splits on
    silence gaps, so a wav with multiple speech blocks separated by true
    silence yields multiple raw segments -- needed to drive ``build_ipus``
    into producing more than one IPU per channel."""

    def __init__(self, frame_sec: float = 0.01, threshold: float = 1e-6):
        self.frame_sec = frame_sec
        self.threshold = threshold

    def __call__(self, wav, sr):
        frame = max(1, int(round(self.frame_sec * sr)))
        out = []
        in_speech = False
        start = 0
        n = len(wav)
        for i in range(0, n, frame):
            block = wav[i : i + frame]
            active = bool(np.max(np.abs(block)) > self.threshold)
            if active and not in_speech:
                start, in_speech = i, True
            elif not active and in_speech:
                out.append((start / sr, i / sr))
                in_speech = False
        if in_speech:
            out.append((start / sr, n / sr))
        return out


class KeyedFakeEmbedder:
    """Deterministic fake embedder: looks up a vector by the EXACT sample
    content it's called with (see module docstring). ``register`` returns the
    array it was given so callers can write it straight to a wav file."""

    def __init__(self):
        self._table: dict[tuple, np.ndarray] = {}

    @staticmethod
    def _key(wav) -> tuple:
        return tuple(np.round(np.asarray(wav, dtype=np.float64), 6).tolist())

    def register(self, wav: np.ndarray, vector) -> np.ndarray:
        arr = np.asarray(wav, dtype=np.float32)
        self._table[self._key(arr)] = np.asarray(vector, dtype=np.float64)
        return arr

    def __call__(self, wav, sr):
        key = self._key(wav)
        if key not in self._table:
            raise KeyError(
                f"no fake embedding registered for snippet of len {len(wav)}"
            )
        return self._table[key]


def _write_wav_exact(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def _block(duration_s: float, amplitude: float, sr: int) -> np.ndarray:
    return np.full(int(round(duration_s * sr)), amplitude, dtype=np.float32)


# --------------------------------------------------------------------------- #
# pure math helpers
# --------------------------------------------------------------------------- #
class TestCosine:
    def test_identical_vectors_are_similarity_one(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_similarity_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_are_similarity_minus_one(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_scale_invariant(self):
        assert _cosine([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)

    def test_zero_vector_returns_zero_not_nan(self):
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


class TestLeastSquaresSlope:
    def test_perfect_line_recovers_exact_slope(self):
        # y = 2x - 1 -> slope 2
        slope = _least_squares_slope([0, 1, 2, 3], [-1.0, 1.0, 3.0, 5.0])
        assert slope == pytest.approx(2.0)

    def test_flat_line_has_zero_slope(self):
        slope = _least_squares_slope([0, 1, 2], [0.5, 0.5, 0.5])
        assert slope == pytest.approx(0.0)

    def test_decreasing_curve_has_negative_slope(self):
        slope = _least_squares_slope([0, 1, 2], [1.0, 0.5, 0.0])
        assert slope == pytest.approx(-0.5)


class TestPercentile:
    def test_empty_returns_none(self):
        assert _percentile([], 50) is None

    def test_median_of_odd_length_list(self):
        assert _percentile([1.0, 3.0, 2.0], 50) == pytest.approx(2.0)

    def test_p90_matches_numpy_reference(self):
        values = [float(v) for v in range(1, 11)]
        assert _percentile(values, 90) == pytest.approx(np.percentile(values, 90))


# --------------------------------------------------------------------------- #
# _clip_turns_to_region: GT turns (window-relative, per src/inference.py's
# _turn_spans) -> generated-region-relative intervals per channel.
# --------------------------------------------------------------------------- #
class TestClipTurnsToRegion:
    def test_turn_entirely_after_boundary_is_shifted_not_clipped(self):
        turns = [{"channel": 0, "text": "hi", "start": 5.0, "end": 6.0}]
        out = _clip_turns_to_region(turns, boundary_sec=5.0, region_duration=10.0)
        assert out == {0: [(0.0, 1.0)]}

    def test_turn_straddling_the_boundary_is_clipped_to_zero(self):
        # starts before the boundary (in the prompt), ends after it: only the
        # portion inside the generated region (which is all we have audio
        # for) should survive.
        turns = [{"channel": 0, "text": "hi", "start": 4.0, "end": 6.0}]
        out = _clip_turns_to_region(turns, boundary_sec=5.0, region_duration=10.0)
        assert out == {0: [(0.0, 1.0)]}

    def test_turn_entirely_before_boundary_is_dropped(self):
        turns = [{"channel": 0, "text": "hi", "start": 1.0, "end": 2.0}]
        out = _clip_turns_to_region(turns, boundary_sec=5.0, region_duration=10.0)
        assert out == {}

    def test_turn_extending_past_region_end_is_clipped_to_region_duration(self):
        turns = [{"channel": 0, "text": "hi", "start": 12.0, "end": 20.0}]
        out = _clip_turns_to_region(turns, boundary_sec=5.0, region_duration=10.0)
        assert out == {0: [(7.0, 10.0)]}

    def test_multiple_channels_grouped_separately(self):
        turns = [
            {"channel": 0, "text": "a", "start": 5.0, "end": 6.0},
            {"channel": 1, "text": "b", "start": 6.5, "end": 7.5},
        ]
        out = _clip_turns_to_region(turns, boundary_sec=5.0, region_duration=10.0)
        assert out == {0: [(0.0, 1.0)], 1: [(1.5, 2.5)]}


class TestRegionSumsq:
    def test_constant_amplitude_region_matches_hand_computed_power(self):
        sr = 1000
        wav = np.full(sr, 0.5, dtype=np.float32)  # 1s of constant 0.5
        sumsq, frames = _region_sumsq(wav, sr, [(0.0, 1.0)])
        assert frames == sr
        assert sumsq == pytest.approx(sr * 0.25)

    def test_region_past_end_of_wav_is_clamped(self):
        sr = 1000
        wav = np.full(sr, 1.0, dtype=np.float32)
        sumsq, frames = _region_sumsq(wav, sr, [(0.5, 2.0)])
        assert frames == 500
        assert sumsq == pytest.approx(500.0)

    def test_empty_regions_yield_zero(self):
        wav = np.zeros(100, dtype=np.float32)
        sumsq, frames = _region_sumsq(wav, 100, [])
        assert (sumsq, frames) == (0.0, 0)


# --------------------------------------------------------------------------- #
# ChannelFeatures-level pure aggregation: _sim_o, _consistency, _drift,
# _confusion_pairs operate on already-computed embeddings (constructed
# directly here), independent of audio I/O / VAD.
# --------------------------------------------------------------------------- #
def _feat(gen=None, prompt=None, per_ipu=()):
    return ChannelFeatures(
        ipus=[(float(i), float(i) + 1.0) for i in range(len(per_ipu))],
        gen_embedding=None if gen is None else np.asarray(gen, dtype=np.float64),
        prompt_embedding=(
            None if prompt is None else np.asarray(prompt, dtype=np.float64)
        ),
        per_ipu_embeddings=[
            None if e is None else np.asarray(e, dtype=np.float64) for e in per_ipu
        ],
    )


class TestSimO:
    def test_matching_gen_and_prompt_embedding_is_similarity_one(self):
        assert _sim_o(_feat(gen=[1.0, 0.0], prompt=[1.0, 0.0])) == pytest.approx(1.0)

    def test_orthogonal_gen_and_prompt_embedding_is_zero(self):
        assert _sim_o(_feat(gen=[1.0, 0.0], prompt=[0.0, 1.0])) == pytest.approx(0.0)

    def test_missing_gen_embedding_returns_none(self):
        assert _sim_o(_feat(gen=None, prompt=[1.0, 0.0])) is None

    def test_missing_prompt_embedding_returns_none(self):
        assert _sim_o(_feat(gen=[1.0, 0.0], prompt=None)) is None


class TestConsistency:
    def test_identical_ipu_embeddings_give_consistency_one(self):
        feat = _feat(per_ipu=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        assert _consistency(feat) == pytest.approx(1.0)

    def test_two_orthogonal_ipus_give_consistency_zero(self):
        feat = _feat(per_ipu=[[1.0, 0.0], [0.0, 1.0]])
        assert _consistency(feat) == pytest.approx(0.0)

    def test_mean_pairwise_over_three_ipus(self):
        # pairs: (0,1)->0, (0,2)->1, (1,2)->0 => mean 1/3
        feat = _feat(per_ipu=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        assert _consistency(feat) == pytest.approx(1.0 / 3.0)

    def test_single_ipu_returns_none(self):
        assert _consistency(_feat(per_ipu=[[1.0, 0.0]])) is None

    def test_no_ipus_returns_none(self):
        assert _consistency(_feat(per_ipu=[])) is None

    def test_skipped_too_short_ipus_are_excluded_not_zero(self):
        # a None entry (skipped by the embed_min_sec floor) must not be
        # treated as a zero-similarity point; it should just be ignored.
        feat = _feat(per_ipu=[[1.0, 0.0], None, [1.0, 0.0]])
        assert _consistency(feat) == pytest.approx(1.0)


class TestDrift:
    def test_perfect_linear_decay_recovers_exact_slope_and_curve(self):
        # cos-to-prompt at index 0,1,2 = 1, 0.5, 0 (unit vectors at 0/60/90deg)
        prompt = [1.0, 0.0]
        e0 = [1.0, 0.0]
        e1 = [0.5, math.sqrt(3) / 2]
        e2 = [0.0, 1.0]
        feat = _feat(prompt=prompt, per_ipu=[e0, e1, e2])
        curve, slope = _drift(feat)
        assert curve == pytest.approx([1.0, 0.5, 0.0], abs=1e-6)
        assert slope == pytest.approx(-0.5, abs=1e-6)

    def test_missing_prompt_embedding_yields_all_none_curve_and_no_slope(self):
        feat = _feat(prompt=None, per_ipu=[[1.0, 0.0], [0.0, 1.0]])
        curve, slope = _drift(feat)
        assert curve == [None, None]
        assert slope is None

    def test_single_defined_point_has_no_slope(self):
        feat = _feat(prompt=[1.0, 0.0], per_ipu=[[1.0, 0.0]])
        curve, slope = _drift(feat)
        assert curve == pytest.approx([1.0])
        assert slope is None

    def test_skipped_ipu_leaves_a_none_hole_in_the_curve_at_its_original_index(self):
        feat = _feat(prompt=[1.0, 0.0], per_ipu=[[1.0, 0.0], None, [0.0, 1.0]])
        curve, slope = _drift(feat)
        assert curve[1] is None
        assert curve[0] == pytest.approx(1.0)
        assert curve[2] == pytest.approx(0.0)
        # slope fit over the two defined points at original indices 0 and 2.
        assert slope == pytest.approx((0.0 - 1.0) / (2 - 0))


class TestConfusionPairs:
    def test_two_channels_each_ordered_pair_scored(self):
        features = [
            _feat(gen=[1.0, 0.0], prompt=[1.0, 0.0]),
            _feat(gen=[0.0, 1.0], prompt=[0.0, 1.0]),
        ]
        pairs = _confusion_pairs(features)
        by_pair = {(p["gen_channel"], p["prompt_channel"]): p["cosine"] for p in pairs}
        assert set(by_pair) == {(0, 1), (1, 0)}
        assert by_pair[(0, 1)] == pytest.approx(0.0)  # gen ch0 vs prompt ch1
        assert by_pair[(1, 0)] == pytest.approx(0.0)

    def test_missing_embedding_skips_that_pair_only(self):
        features = [
            _feat(gen=None, prompt=[1.0, 0.0]),
            _feat(gen=[0.0, 1.0], prompt=[0.0, 1.0]),
        ]
        pairs = _confusion_pairs(features)
        # (0,1) needs gen ch0 (missing) -> skipped; (1,0) needs gen ch1 +
        # prompt ch0, both present.
        assert [(p["gen_channel"], p["prompt_channel"]) for p in pairs] == [(1, 0)]

    def test_single_channel_has_no_pairs(self):
        assert _confusion_pairs([_feat(gen=[1.0, 0.0], prompt=[1.0, 0.0])]) == []


# --------------------------------------------------------------------------- #
# _channel_features: VAD + IPU construction + embed_min_sec floor, driven
# through the metric instance (constructor knobs matter here).
# --------------------------------------------------------------------------- #
class TestChannelFeatures:
    SR = 16000

    def test_concatenation_pools_all_ipus_even_a_short_one_excluded_from_per_ipu(
        self, tmp_path
    ):
        # Two IPUs: a normal one (0.5s) and a short one (0.05s, below the
        # embed_min_sec floor). The short IPU must still contribute its
        # audio to the CONCATENATED embedding input (SIM-o/confusion), but
        # must be skipped as an individually embedded point (consistency/
        # drift) -- the documented asymmetry.
        sr = self.SR
        long_ipu = _block(0.5, 0.11, sr)
        short_ipu = _block(0.05, 0.22, sr)
        wav = np.concatenate(
            [long_ipu, np.zeros(int(0.3 * sr), dtype=np.float32), short_ipu]
        )
        gen_path = tmp_path / "gen.wav"
        _write_wav_exact(gen_path, wav, sr)
        prompt_wav = _block(0.5, 0.33, sr)
        prompt_path = tmp_path / "prompt.wav"
        _write_wav_exact(prompt_path, prompt_wav, sr)

        embedder = KeyedFakeEmbedder()
        # register() returns the audio array it was given (not the vector),
        # so the exact concatenation the metric builds can be re-registered
        # elsewhere if needed; here we only need the registration side effect.
        embedder.register(np.concatenate([long_ipu, short_ipu]), [1.0, 0.0])
        embedder.register(long_ipu, [0.0, 1.0])  # per-IPU embedding of the long IPU
        embedder.register(prompt_wav, [1.0, 1.0])

        vad = _ListVAD([(0.0, 0.5), (0.8, 0.85)])
        metric = SpeakerDynamicsMetric(
            embedder=embedder,
            vad=vad,
            min_silence=0.2,
            min_speech=0.0,
            pad=0.0,
            embed_min_sec=0.1,
        )

        feat = metric._channel_features(gen_path, prompt_path)

        assert feat.ipus == [(0.0, 0.5), (0.8, 0.85)]
        assert feat.gen_embedding.tolist() == pytest.approx([1.0, 0.0])
        assert feat.per_ipu_embeddings[0].tolist() == pytest.approx([0.0, 1.0])
        assert feat.per_ipu_embeddings[1] is None  # short IPU: below embed_min_sec
        assert feat.prompt_embedding.tolist() == pytest.approx([1.0, 1.0])

    def test_prompt_below_floor_is_none(self, tmp_path):
        sr = self.SR
        gen_wav = _block(0.5, 0.11, sr)
        gen_path = tmp_path / "gen.wav"
        _write_wav_exact(gen_path, gen_wav, sr)
        short_prompt = _block(0.05, 0.5, sr)
        prompt_path = tmp_path / "prompt.wav"
        _write_wav_exact(prompt_path, short_prompt, sr)

        embedder = KeyedFakeEmbedder()
        embedder.register(gen_wav, [1.0, 0.0])

        metric = SpeakerDynamicsMetric(
            embedder=embedder,
            vad=_ListVAD([(0.0, 0.5)]),
            embed_min_sec=0.1,
        )
        feat = metric._channel_features(gen_path, prompt_path)
        assert feat.prompt_embedding is None
        assert feat.gen_embedding is not None

    def test_no_ipus_yields_none_gen_embedding_and_embedder_never_called_for_it(
        self, tmp_path
    ):
        sr = self.SR
        gen_path = tmp_path / "gen.wav"
        _write_wav_exact(gen_path, np.zeros(sr, dtype=np.float32), sr)
        prompt_path = tmp_path / "prompt.wav"
        prompt_wav = _block(0.5, 0.2, sr)
        _write_wav_exact(prompt_path, prompt_wav, sr)

        embedder = KeyedFakeEmbedder()
        embedder.register(prompt_wav, [1.0, 0.0])

        metric = SpeakerDynamicsMetric(embedder=embedder, vad=_ListVAD([]))
        feat = metric._channel_features(gen_path, prompt_path)
        assert feat.ipus == []
        assert feat.gen_embedding is None
        assert feat.per_ipu_embeddings == []


# --------------------------------------------------------------------------- #
# bleed dB: synthetic two-channel audio with a known injected leak, solo
# regions derived from GT turns (not VAD).
# --------------------------------------------------------------------------- #
class TestBleedViaFullWindow:
    """Exercised through the real __call__ so the GT-turn -> solo-region ->
    energy-ratio pipeline is proven end to end, not just its pieces."""

    SR = 24000

    def _metric(self):
        # VAD/embedder are irrelevant to bleed but still required by the
        # metric; a trivial always-empty VAD keeps sim_o/consistency/drift/
        # confusion all None without needing a real embedder.
        return SpeakerDynamicsMetric(embedder=KeyedFakeEmbedder(), vad=_ListVAD([]))

    def test_known_leak_produces_hand_computed_bleed_db(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        wid = "sess_w00000"

        # Channel 0 ("own", amplitude 1.0) is scripted solo for the whole
        # 2s generated region; channel 1 ("leak", amplitude 0.1) has no
        # scripted turns there at all but its generated audio still has
        # energy -- exactly the failure mode bleed dB measures.
        own = _block(2.0, 1.0, sr)
        leak = _block(2.0, 0.1, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch0.wav", own, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch1.wav", leak, sr)
        _write_wav_exact(
            test_dir / "prompt" / f"{wid}_ch0.wav", np.zeros(1, dtype=np.float32), sr
        )
        _write_wav_exact(
            test_dir / "prompt" / f"{wid}_ch1.wav", np.zeros(1, dtype=np.float32), sr
        )
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch0.wav", own, sr)
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch1.wav", leak, sr)

        boundary = 5.0
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + 2.0,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"prompt/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"prompt/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "",
                },
            ],
            # channel 0 speaks solo for the entire generated region
            # (window-relative time, matching src/inference.py's _turn_spans
            # convention: turns carry the SAME window-relative clock as
            # prompt_boundary_sec, not region-relative time).
            "turns": [
                {"channel": 0, "text": "", "start": boundary, "end": boundary + 2.0},
            ],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")
        (test_dir / "meta.scp").write_text(f"{wid} meta/{wid}.json\n", encoding="utf-8")

        metric = self._metric()
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        # guard default 0.2s trims 0.2s off each edge of the 2s solo span ->
        # 1.6s of measured solo region, well inside both constant blocks.
        expected_db = 10 * math.log10((0.1**2) / (1.0**2))
        assert summary["bleed_db_p50"] == pytest.approx(expected_db, abs=1e-6)
        assert summary["bleed_db_p90"] == pytest.approx(expected_db, abs=1e-6)

        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_dynamics"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert len(record["bleed_pairs"]) == 1
        pair = record["bleed_pairs"][0]
        assert pair["gen_channel"] == 1
        assert pair["solo_channel"] == 0
        assert pair["bleed_db"] == pytest.approx(expected_db, abs=1e-6)
        # ch0<-ch1: channel 1 never has a solo span (it has zero scripted
        # turns at all) -> skipped, not silently dropped.
        assert [0, 1] in record["bleed_skipped_pairs"]

    def test_no_solo_region_skips_the_pair_and_counts_it(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        wid = "sess_w00000"
        both = _block(2.0, 0.3, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch0.wav", both, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch1.wav", both, sr)
        _write_wav_exact(
            test_dir / "prompt" / f"{wid}_ch0.wav", np.zeros(1, dtype=np.float32), sr
        )
        _write_wav_exact(
            test_dir / "prompt" / f"{wid}_ch1.wav", np.zeros(1, dtype=np.float32), sr
        )
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch0.wav", both, sr)
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch1.wav", both, sr)

        boundary = 5.0
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + 2.0,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"prompt/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"prompt/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "",
                },
            ],
            # both channels active for the whole region -> no solo span for
            # either channel.
            "turns": [
                {"channel": 0, "text": "", "start": boundary, "end": boundary + 2.0},
                {"channel": 1, "text": "", "start": boundary, "end": boundary + 2.0},
            ],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")
        (test_dir / "meta.scp").write_text(f"{wid} meta/{wid}.json\n", encoding="utf-8")

        metric = self._metric()
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        assert summary["bleed_db_p50"] == pytest.approx(0.0)
        assert summary["bleed_db_p90"] == pytest.approx(0.0)

        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_dynamics"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert record["bleed_pairs"] == []
        assert sorted(record["bleed_skipped_pairs"]) == [[0, 1], [1, 0]]


# --------------------------------------------------------------------------- #
# backend laziness: constructing the real defaults must never import their
# heavy package; only the first call may.
# --------------------------------------------------------------------------- #
class TestBackendLaziness:
    def test_wavlm_sv_embedder_construction_does_not_import_transformers(
        self, monkeypatch
    ):
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "transformers" or name.startswith("transformers."):
                raise AssertionError("transformers imported before first call")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)
        embedder = WavLMSVEmbedder()
        assert embedder._model is None

    def test_metric_construction_with_all_real_defaults_does_not_touch_network(self):
        metric = SpeakerDynamicsMetric()
        assert isinstance(metric.embedder, WavLMSVEmbedder)
        assert metric.embedder.model_tag == "microsoft/wavlm-base-plus-sv"

    def test_embedder_rejects_non_16k_audio_without_loading_the_model(self):
        embedder = WavLMSVEmbedder()
        with pytest.raises(ValueError, match="16000"):
            embedder(np.zeros(100, dtype=np.float32), 8000)
        assert embedder._model is None


# --------------------------------------------------------------------------- #
# full __call__ round trip: JSONL/summary artifacts, meta-relative path
# resolution, and the documented summary-key set.
# --------------------------------------------------------------------------- #
class TestCallRoundTrip:
    SR = 16000

    def _build_window(self, test_dir: Path, wid: str, boundary: float = 5.0) -> None:
        sr = self.SR
        for ch in (0, 1):
            _write_wav_exact(
                test_dir / "wav" / f"{wid}_ch{ch}.wav",
                np.zeros(int(0.3 * sr), dtype=np.float32),
                sr,
            )
            _write_wav_exact(
                test_dir / "prompt" / f"{wid}_ch{ch}.wav",
                np.zeros(1, dtype=np.float32),
                sr,
            )
            _write_wav_exact(
                test_dir / "gt" / f"{wid}_ch{ch}.wav",
                np.zeros(int(0.3 * sr), dtype=np.float32),
                sr,
            )
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + 0.3,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"prompt/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"prompt/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "",
                },
            ],
            "turns": [],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_writes_jsonl_and_summary_with_the_documented_keys(self, tmp_path):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        self._build_window(test_dir, "sess_w00000")
        (test_dir / "meta.scp").write_text(
            "sess_w00000 meta/sess_w00000.json\n", encoding="utf-8"
        )

        metric = SpeakerDynamicsMetric(embedder=KeyedFakeEmbedder(), vad=_ListVAD([]))
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        assert set(summary) == {
            "sim_o_mean",
            "sim_consistency",
            "sim_drift_slope",
            "confusion_mean",
            "bleed_db_p50",
            "bleed_db_p90",
        }
        assert all(isinstance(v, float) for v in summary.values())

        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_dynamics"
        lines = (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["window_id"] == "sess_w00000"

        on_disk_summary = json.loads((scoring_dir / "summary.json").read_text("utf-8"))
        assert on_disk_summary == summary

    def test_meta_relative_paths_resolve_against_the_test_dir(self, tmp_path):
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        sr = self.SR
        wid = "sess_w00000"
        gen0 = _block(0.5, 0.4, sr)
        prompt0 = _block(0.5, 0.4, sr)
        for ch, gen in ((0, gen0), (1, np.zeros(int(0.3 * sr), dtype=np.float32))):
            _write_wav_exact(test_dir / "wav" / f"{wid}_ch{ch}.wav", gen, sr)
            _write_wav_exact(
                test_dir / "gt" / f"{wid}_ch{ch}.wav",
                np.zeros(int(0.3 * sr), dtype=np.float32),
                sr,
            )
        _write_wav_exact(test_dir / "prompt" / f"{wid}_ch0.wav", prompt0, sr)
        _write_wav_exact(
            test_dir / "prompt" / f"{wid}_ch1.wav", np.zeros(1, dtype=np.float32), sr
        )
        boundary = 5.0
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + 0.3,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"prompt/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"prompt/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "",
                },
            ],
            "turns": [],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")
        (test_dir / "meta.scp").write_text(f"{wid} meta/{wid}.json\n", encoding="utf-8")

        embedder = KeyedFakeEmbedder()
        embedder.register(gen0, [1.0, 0.0])
        embedder.register(prompt0, [1.0, 0.0])
        metric = SpeakerDynamicsMetric(
            embedder=embedder, vad=_EnergyVAD(), embed_min_sec=0.1
        )
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)
        # ch0 gen matches its own prompt exactly -> sim_o=1 for ch0; ch1 has
        # no speech (empty gen -> no IPUs -> sim_o None), so the per-window
        # mean is over ch0 alone -> window mean 1.0 -> summary 1.0.
        assert summary["sim_o_mean"] == pytest.approx(1.0)

    def test_multi_ipu_channel_produces_drift_curve_and_consistency_in_jsonl(
        self, tmp_path
    ):
        """Closes the gap the other __call__ tests leave: they all give the
        generated channel 0 or 1 IPU, so sim_consistency/sim_drift_slope are
        always None -> always the 0.0 fallback, and the JSONL's drift_curve
        never has more than one point. Here channel 0 gets TWO well-separated
        IPUs with known, non-parallel embeddings, so every one of those
        fields has a hand-computable, non-fallback value."""
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        sr = self.SR
        wid = "sess_w00000"

        ipu0 = _block(0.5, 0.11, sr)
        gap = np.zeros(int(0.3 * sr), dtype=np.float32)
        ipu1 = _block(0.5, 0.22, sr)
        gen0 = np.concatenate([ipu0, gap, ipu1])
        prompt0 = _block(0.5, 0.33, sr)
        silent = np.zeros(int(0.3 * sr), dtype=np.float32)

        _write_wav_exact(test_dir / "wav" / f"{wid}_ch0.wav", gen0, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch1.wav", silent, sr)
        _write_wav_exact(test_dir / "prompt" / f"{wid}_ch0.wav", prompt0, sr)
        _write_wav_exact(
            test_dir / "prompt" / f"{wid}_ch1.wav", np.zeros(1, dtype=np.float32), sr
        )
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch0.wav", gen0, sr)
        _write_wav_exact(test_dir / "gt" / f"{wid}_ch1.wav", silent, sr)

        boundary = 5.0
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + gen0.shape[0] / sr,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"prompt/{wid}_ch0.wav",
                    "gt_wav": f"gt/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"prompt/{wid}_ch1.wav",
                    "gt_wav": f"gt/{wid}_ch1.wav",
                    "ref_text": "",
                },
            ],
            "turns": [],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")
        (test_dir / "meta.scp").write_text(f"{wid} meta/{wid}.json\n", encoding="utf-8")

        # prompt p=[1,0]; IPU embeddings e0=[1,0] (cos-to-prompt 1.0),
        # e1=[0.6,0.8] (cos-to-prompt 0.6, both unit vectors) -> consistency
        # (their single pairwise cosine) = 0.6; drift curve [1.0, 0.6],
        # least-squares slope over x=[0,1] = 0.6 - 1.0 = -0.4.
        embedder = KeyedFakeEmbedder()
        embedder.register(ipu0, [1.0, 0.0])
        embedder.register(ipu1, [0.6, 0.8])
        embedder.register(np.concatenate([ipu0, ipu1]), [1.0, 0.0])  # gen_embedding
        embedder.register(prompt0, [1.0, 0.0])

        metric = SpeakerDynamicsMetric(embedder=embedder, vad=_FrameEnergyVAD())
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_dynamics"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert record["consistency"][0] == pytest.approx(0.6)
        assert record["drift_curve"][0] == pytest.approx([1.0, 0.6], abs=1e-6)
        assert record["drift_slope"][0] == pytest.approx(-0.4, abs=1e-6)
        # channel 1 never speaks -> no IPUs -> no consistency/drift value.
        assert record["consistency"][1] is None
        assert record["drift_curve"][1] == []
        assert record["drift_slope"][1] is None

        # window-level mean is over channel 0 alone (channel 1 contributes
        # no value, not a fabricated 0) -> summary equals channel 0's value.
        assert summary["sim_consistency"] == pytest.approx(0.6)
        assert summary["sim_drift_slope"] == pytest.approx(-0.4, abs=1e-6)


# --------------------------------------------------------------------------- #
# conf/metrics.yaml wiring: the binding constraint that the shipped config
# instantiates every metric offline with its real (lazy) defaults, i.e.
# constructing SpeakerDynamicsMetric() from the config never downloads.
# --------------------------------------------------------------------------- #
class TestMetricsConfigInstantiatesOffline:
    def test_speaker_dynamics_metric_entry_instantiates_without_network(
        self, monkeypatch
    ):
        import builtins

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

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "transformers" or name.startswith("transformers."):
                raise AssertionError("transformers imported while instantiating config")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)

        speaker_entries = [
            entry
            for entry in metrics_config.metrics
            if entry.metric._target_.endswith("SpeakerDynamicsMetric")
        ]
        assert len(speaker_entries) == 1
        metric = instantiate(speaker_entries[0].metric)
        assert isinstance(metric, SpeakerDynamicsMetric)
        assert isinstance(metric.embedder, WavLMSVEmbedder)
        assert metric.embedder._model is None

    def test_every_configured_metric_instantiates_without_network(self, monkeypatch):
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
        metrics = [instantiate(entry.metric) for entry in metrics_config.metrics]
        assert len(metrics) == 2
        assert any(isinstance(m, SpeakerDynamicsMetric) for m in metrics)


# --------------------------------------------------------------------------- #
# asset-gated real-backend smoke
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
class TestRealBackendSmoke:
    def test_real_embedder_returns_a_1d_vector_for_silence(self):
        embedder = WavLMSVEmbedder()
        silence = np.zeros(16000, dtype=np.float32)
        try:
            emb = embedder(silence, 16000)
        except OSError:
            pytest.skip("wavlm-base-plus-sv weights not available offline")
        assert emb.ndim == 1
        assert emb.shape[0] > 0
