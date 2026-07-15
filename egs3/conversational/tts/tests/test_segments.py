"""``src/metrics/segments.py`` tests: wav load/resample, the injectable VAD
wrapper, dGSLM IPU construction, and interval-helper re-export identity.

CPU-only, no network: the VAD wrapper is exercised with a deterministic
energy-threshold fake backend defined here (never the real silero download,
which is only reachable via ``SileroVADBackend`` and is asset-gated / lazy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.local import crosstalk_report
from egs3.conversational.tts.src.metrics import segments


# --------------------------------------------------------------------------- #
# wav loading + resampling
# --------------------------------------------------------------------------- #
def _write_tone(
    path: Path, sr: int, duration_s: float, freq: float = 300.0
) -> np.ndarray:
    n = int(round(duration_s * sr))
    t = np.arange(n) / sr
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sr)
    return data


class TestLoadWav:
    def test_native_rate_round_trips(self, tmp_path):
        sr = 24000
        path = tmp_path / "x.wav"
        data = _write_tone(path, sr, 0.5)

        wav, out_sr = segments.load_wav(path)

        assert out_sr == sr
        assert wav.shape[0] == data.shape[0]
        np.testing.assert_allclose(wav, data, atol=1e-4)

    def test_resamples_to_target_rate(self, tmp_path):
        sr = 24000
        path = tmp_path / "x.wav"
        data = _write_tone(path, sr, 0.5)

        wav, out_sr = segments.load_wav(path, target_sr=16000)

        assert out_sr == 16000
        expected_len = round(data.shape[0] * 16000 / sr)
        assert abs(wav.shape[0] - expected_len) <= 2

    def test_target_sr_equal_to_native_is_a_noop(self, tmp_path):
        sr = 16000
        path = tmp_path / "x.wav"
        data = _write_tone(path, sr, 0.2)

        wav, out_sr = segments.load_wav(path, target_sr=16000)

        assert out_sr == 16000
        assert wav.shape[0] == data.shape[0]

    def test_stereo_file_is_downmixed_to_mono(self, tmp_path):
        sr = 16000
        n = 800
        stereo = np.stack(
            [np.full(n, 0.2, dtype=np.float32), np.full(n, 0.6, dtype=np.float32)],
            axis=1,
        )
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, sr)

        wav, out_sr = segments.load_wav(path)

        assert out_sr == sr
        assert wav.ndim == 1
        assert wav.shape[0] == n
        np.testing.assert_allclose(wav, np.full(n, 0.4, dtype=np.float32), atol=1e-4)


# --------------------------------------------------------------------------- #
# re-exported interval helpers: identical objects, not reimplementations
# --------------------------------------------------------------------------- #
class TestReexportedIntervalHelpers:
    def test_merge_intervals_is_the_same_function(self):
        assert segments.merge_intervals is crosstalk_report.merge_intervals

    def test_subtract_intervals_is_the_same_function(self):
        assert segments.subtract_intervals is crosstalk_report.subtract_intervals

    def test_solo_regions_is_the_same_function(self):
        assert segments.solo_regions is crosstalk_report.solo_regions


# --------------------------------------------------------------------------- #
# build_ipus: the dGSLM 200ms rule, as a pure function over intervals
# --------------------------------------------------------------------------- #
def _approx_intervals(actual, expected):
    assert len(actual) == len(expected)
    for (a0, a1), (e0, e1) in zip(actual, expected):
        assert a0 == pytest.approx(e0, abs=1e-6)
        assert a1 == pytest.approx(e1, abs=1e-6)


class TestBuildIpus:
    def test_merges_segments_within_min_silence(self):
        ipus = segments.build_ipus([(0.0, 0.4), (0.55, 0.8)], min_silence=0.2)
        _approx_intervals(ipus, [(0.0, 0.8)])

    def test_keeps_segments_separated_by_more_than_min_silence(self):
        ipus = segments.build_ipus([(0.0, 0.4), (0.7, 0.8)], min_silence=0.2)
        _approx_intervals(ipus, [(0.0, 0.4), (0.7, 0.8)])

    def test_gap_exactly_min_silence_still_merges(self):
        # "separated by MORE than min_silence are distinct" -> gap == min_silence
        # is not "more than", so it merges.
        ipus = segments.build_ipus([(0.0, 0.2), (0.4, 0.6)], min_silence=0.2)
        _approx_intervals(ipus, [(0.0, 0.6)])

    def test_min_speech_drops_short_ipus(self):
        ipus = segments.build_ipus(
            [(0.0, 0.05), (2.0, 2.5)], min_silence=0.2, min_speech=0.1
        )
        _approx_intervals(ipus, [(2.0, 2.5)])

    def test_min_speech_default_keeps_short_ipus(self):
        ipus = segments.build_ipus([(0.0, 0.05), (2.0, 2.5)], min_silence=0.2)
        _approx_intervals(ipus, [(0.0, 0.05), (2.0, 2.5)])

    def test_pad_extends_edges_and_clamps_to_zero(self):
        ipus = segments.build_ipus([(0.05, 0.2)], min_silence=0.2, pad=0.1)
        _approx_intervals(ipus, [(0.0, 0.3)])

    def test_pad_clamps_to_total_duration(self):
        ipus = segments.build_ipus(
            [(1.0, 1.2)], min_silence=0.2, pad=0.1, total_duration=1.22
        )
        _approx_intervals(ipus, [(0.9, 1.22)])

    def test_padding_that_closes_a_gap_remerges(self):
        # Raw gap (0.4) exceeds min_silence (0.05) so these start distinct;
        # padding by 0.25 on each edge closes the gap and must remerge them
        # into one IPU, not leave two overlapping intervals.
        ipus = segments.build_ipus([(0.0, 0.1), (0.5, 0.6)], min_silence=0.05, pad=0.25)
        _approx_intervals(ipus, [(0.0, 0.85)])

    def test_empty_input_returns_empty(self):
        assert segments.build_ipus([]) == []


# --------------------------------------------------------------------------- #
# VAD wrapper driven by a deterministic energy backend, over synthetic audio
# --------------------------------------------------------------------------- #
class EnergyVADBackend:
    """Frame-wise RMS threshold: deterministic, no network, test-only."""

    def __init__(self, frame_sec: float = 0.01, threshold: float = 0.1):
        self.frame_sec = frame_sec
        self.threshold = threshold

    def __call__(self, wav: np.ndarray, sr: int):
        frame = max(1, int(round(self.frame_sec * sr)))
        out = []
        in_speech = False
        start = 0
        n = len(wav)
        for i in range(0, n, frame):
            block = wav[i : i + frame]
            rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
            active = rms > self.threshold
            if active and not in_speech:
                start, in_speech = i, True
            elif not active and in_speech:
                out.append((start / sr, i / sr))
                in_speech = False
        if in_speech:
            out.append((start / sr, n / sr))
        return out


def _block(duration_s: float, amplitude: float, sr: int) -> np.ndarray:
    return np.full(int(round(duration_s * sr)), amplitude, dtype=np.float32)


class TestVadWithSyntheticAudio:
    """Known on/off speech pattern -> exact expected raw segments and IPUs."""

    SR = 16000

    def _wav(self) -> np.ndarray:
        sr = self.SR
        return np.concatenate(
            [
                _block(0.10, 0.5, sr),  # speech   0.00-0.10
                _block(0.15, 0.0, sr),  # silence  0.10-0.25 (150ms < 200ms)
                _block(0.15, 0.5, sr),  # speech   0.25-0.40
                _block(0.30, 0.0, sr),  # silence  0.40-0.70 (300ms > 200ms)
                _block(0.10, 0.5, sr),  # speech   0.70-0.80
                _block(0.20, 0.0, sr),  # trailing silence
            ]
        )

    def test_energy_backend_raw_segments(self):
        wav = self._wav()
        vad = segments.VAD(backend=EnergyVADBackend())
        raw = vad(wav, self.SR)
        _approx_intervals(raw, [(0.0, 0.10), (0.25, 0.40), (0.70, 0.80)])

    def test_dgslm_ipus_from_synthetic_pattern(self):
        wav = self._wav()
        vad = segments.VAD(backend=EnergyVADBackend())
        raw = vad(wav, self.SR)
        ipus = segments.build_ipus(raw, min_silence=0.2)
        # The 150ms gap is absorbed into one IPU; the 300ms gap is not.
        _approx_intervals(ipus, [(0.0, 0.40), (0.70, 0.80)])

    def test_vad_wrapper_casts_backend_output_to_float_tuples(self):
        wav = self._wav()
        vad = segments.VAD(backend=EnergyVADBackend())
        raw = vad(wav, self.SR)
        for start, end in raw:
            assert isinstance(start, float)
            assert isinstance(end, float)


# --------------------------------------------------------------------------- #
# SileroVADBackend: lazy by construction (no network at import/construction)
# --------------------------------------------------------------------------- #
class TestSileroVadIsLazy:
    def test_construction_does_not_touch_torch_hub(self, monkeypatch):
        # Also covers module-import-time laziness: `segments` was already
        # imported at the top of this test module, so if that import had
        # touched the network, collection would have failed before any test
        # ran -- there is no separate attribute to assert on for that.
        import torch

        def _boom(*args, **kwargs):
            raise AssertionError("torch.hub.load must not run before first call")

        monkeypatch.setattr(torch.hub, "load", _boom)

        backend = segments.SileroVADBackend()
        assert backend._model is None
