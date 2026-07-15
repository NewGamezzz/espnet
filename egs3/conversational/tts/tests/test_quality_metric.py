"""``ChannelQualityMetric`` (``src/metrics/quality.py``) tests.

Fake MOS backend / fake VAD, CPU-only, no network: covers the
speech-duration-weighted channel score (with unequal IPU durations so the
weighted mean provably differs from a plain mean), the ``min_ipu_sec``
skip-and-count floor, backend laziness for both the default UTMOS backend
(``torch.hub``-based, see the module docstring's documented deviation from
PLAN-step4.md's "speechmos" wording) and the optional DNSMOS backend (the
genuinely real ``speechmos.dnsmos``), and a full ``__call__`` round trip
against a fabricated ``inference_dir``.

The real ``torch.hub`` UTMOS backend and the real ``speechmos`` DNSMOS
backend are never exercised end to end here -- see
``TorchHubUTMOSBackend``'s docstring for why no live network smoke test is
shipped (the binding "no GPU/model downloads locally" constraint, and no
import-based flag exists to gate a ``torch.hub`` fetch the way
``_HAS_TRANSFORMERS`` gates ``transformers``). The DNSMOS backend IS gated
the standard way (``_HAS_SPEECHMOS``) since it's a genuine importable
package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.quality import (
    ChannelQualityMetric,
    SpeechMOSDNSMOSBackend,
    TorchHubUTMOSBackend,
    _weighted_channel_score,
)
from egs3.conversational.tts.src.metrics.segments import VAD, SileroVADBackend
from egs3.conversational.tts.tests.conftest import _FrameEnergyVAD

try:
    import speechmos  # noqa: F401

    _HAS_SPEECHMOS = True
except ImportError:
    _HAS_SPEECHMOS = False


# --------------------------------------------------------------------------- #
# test-only fakes
# --------------------------------------------------------------------------- #
class KeyedFakeMOSBackend:
    """Deterministic fake MOS backend: looks up a score by the EXACT sample
    content it's called with (mirrors ``test_speaker_metric.py``'s
    ``KeyedFakeEmbedder``)."""

    def __init__(self):
        self._table: dict[tuple, float] = {}

    @staticmethod
    def _key(wav) -> tuple:
        return tuple(np.round(np.asarray(wav, dtype=np.float64), 6).tolist())

    def register(self, wav: np.ndarray, score: float) -> np.ndarray:
        arr = np.asarray(wav, dtype=np.float32)
        self._table[self._key(arr)] = float(score)
        return arr

    def __call__(self, wav, sr):
        key = self._key(wav)
        if key not in self._table:
            raise KeyError(
                f"no fake MOS score registered for snippet of len {len(wav)}"
            )
        return self._table[key]


def _write_wav_exact(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def _block(duration_s: float, amplitude: float, sr: int) -> np.ndarray:
    return np.full(int(round(duration_s * sr)), amplitude, dtype=np.float32)


# --------------------------------------------------------------------------- #
# _weighted_channel_score: the core weighting math, pure numbers
# --------------------------------------------------------------------------- #
class TestWeightedChannelScore:
    def test_unequal_durations_weight_toward_the_longer_ipu(self):
        # plain mean of scores would be (2.0+4.0)/2 = 3.0; the
        # duration-weighted mean skews toward the 3x-longer, higher-scored
        # IPU: (1*2.0 + 3*4.0) / 4 = 14/4 = 3.5 != 3.0.
        ipu_scores = [
            {"duration": 1.0, "score": 2.0},
            {"duration": 3.0, "score": 4.0},
        ]
        weighted = _weighted_channel_score(ipu_scores)
        plain_mean = sum(r["score"] for r in ipu_scores) / len(ipu_scores)
        assert weighted == pytest.approx(3.5)
        assert weighted != pytest.approx(plain_mean)

    def test_equal_durations_weighted_mean_equals_plain_mean(self):
        ipu_scores = [
            {"duration": 1.0, "score": 2.0},
            {"duration": 1.0, "score": 4.0},
        ]
        assert _weighted_channel_score(ipu_scores) == pytest.approx(3.0)

    def test_empty_returns_none(self):
        assert _weighted_channel_score([]) is None

    def test_zero_total_duration_returns_none(self):
        assert _weighted_channel_score([{"duration": 0.0, "score": 5.0}]) is None


# --------------------------------------------------------------------------- #
# _score_channel: min_ipu_sec skip-and-count floor, weighting through the
# real VAD + build_ipus pipeline.
# --------------------------------------------------------------------------- #
class TestScoreChannel:
    SR = 16000

    def test_short_ipu_below_floor_is_skipped_and_counted_not_scored(self, tmp_path):
        sr = self.SR
        long_ipu = _block(0.5, 0.11, sr)
        short_ipu = _block(0.02, 0.22, sr)  # below the 0.05s floor
        gap = np.zeros(int(0.3 * sr), dtype=np.float32)
        wav = np.concatenate([long_ipu, gap, short_ipu])
        path = tmp_path / "gen.wav"
        _write_wav_exact(path, wav, sr)

        backend = KeyedFakeMOSBackend()
        backend.register(long_ipu, 4.0)
        # short_ipu is deliberately NOT registered: if the metric scored it
        # anyway, the fake would raise KeyError and fail the test.

        vad = _FrameEnergyVAD()
        metric = ChannelQualityMetric(mos_backend=backend, vad=vad, min_ipu_sec=0.05)
        record = metric._score_channel(path)

        assert len(record["ipu_scores"]) == 1
        assert record["ipu_scores"][0]["score"] == pytest.approx(4.0)
        assert record["num_ipus_skipped"] == 1
        assert record["channel_score"] == pytest.approx(4.0)

    def test_weighted_score_through_the_real_pipeline(self, tmp_path):
        sr = self.SR
        ipu0 = _block(1.0, 0.11, sr)  # will score 2.0, weight 1.0s
        gap = np.zeros(int(0.3 * sr), dtype=np.float32)
        ipu1 = _block(3.0, 0.22, sr)  # will score 4.0, weight 3.0s
        wav = np.concatenate([ipu0, gap, ipu1])
        path = tmp_path / "gen.wav"
        _write_wav_exact(path, wav, sr)

        backend = KeyedFakeMOSBackend()
        backend.register(ipu0, 2.0)
        backend.register(ipu1, 4.0)

        metric = ChannelQualityMetric(
            mos_backend=backend, vad=_FrameEnergyVAD(), min_ipu_sec=0.05
        )
        record = metric._score_channel(path)

        # (1.0*2.0 + 3.0*4.0) / 4.0 = 3.5, not the plain mean 3.0.
        assert record["channel_score"] == pytest.approx(3.5)
        assert record["num_ipus_skipped"] == 0

    def test_no_ipus_yields_none_channel_score(self, tmp_path):
        sr = self.SR
        path = tmp_path / "silent.wav"
        _write_wav_exact(path, np.zeros(sr, dtype=np.float32), sr)

        metric = ChannelQualityMetric(
            mos_backend=KeyedFakeMOSBackend(), vad=_FrameEnergyVAD()
        )
        record = metric._score_channel(path)
        assert record["ipu_scores"] == []
        assert record["channel_score"] is None


# --------------------------------------------------------------------------- #
# backend laziness
# --------------------------------------------------------------------------- #
class TestBackendLaziness:
    def test_torch_hub_backend_construction_does_not_call_torch_hub_load(
        self, monkeypatch
    ):
        import torch.hub

        def guard(*args, **kwargs):
            raise AssertionError("torch.hub.load called before first __call__")

        monkeypatch.setattr(torch.hub, "load", guard)
        backend = TorchHubUTMOSBackend()
        assert backend._predictor is None

    def test_metric_construction_with_all_real_defaults_does_not_touch_network(self):
        metric = ChannelQualityMetric()
        assert isinstance(metric.mos_backend, TorchHubUTMOSBackend)
        assert metric.mos_backend._predictor is None
        assert isinstance(metric.vad, VAD)
        assert isinstance(metric.vad.backend, SileroVADBackend)
        assert metric.vad.backend._model is None

    def test_dnsmos_backend_construction_does_not_import_speechmos(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "speechmos" or name.startswith("speechmos."):
                raise AssertionError("speechmos imported before first call")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)
        backend = SpeechMOSDNSMOSBackend()
        assert backend._run is None

    def test_dnsmos_disabled_by_default_no_backend_constructed(self):
        metric = ChannelQualityMetric()
        assert metric.enable_dnsmos is False
        assert metric.dnsmos_backend is None

    def test_injecting_dnsmos_backend_without_enabling_it_raises(self):
        # Previously the injected backend was silently discarded (never
        # constructed, never used) when enable_dnsmos stayed False -- a
        # caller who wired one up would get no error and no DNSMOS scoring.
        with pytest.raises(ValueError, match="enable_dnsmos"):
            ChannelQualityMetric(dnsmos_backend=SpeechMOSDNSMOSBackend())

    def test_dnsmos_enabled_constructs_the_default_backend(self):
        metric = ChannelQualityMetric(enable_dnsmos=True)
        assert isinstance(metric.dnsmos_backend, SpeechMOSDNSMOSBackend)

    @pytest.mark.skipif(_HAS_SPEECHMOS, reason="speechmos is installed")
    def test_default_dnsmos_backend_raises_import_error_when_missing(self):
        backend = SpeechMOSDNSMOSBackend()
        with pytest.raises(ImportError, match="speechmos"):
            backend(np.zeros(16000, dtype=np.float32), 16000)


# --------------------------------------------------------------------------- #
# full __call__ round trip
# --------------------------------------------------------------------------- #
class TestCallRoundTrip:
    SR = 16000

    def _write_window(
        self, test_dir: Path, wid: str, ch0_gen: np.ndarray, ch1_gen: np.ndarray
    ) -> None:
        sr = self.SR
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch0.wav", ch0_gen, sr)
        _write_wav_exact(test_dir / "wav" / f"{wid}_ch1.wav", ch1_gen, sr)
        boundary = 5.0
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": 2,
            "prompt_boundary_sec": boundary,
            "prompt_boundary_frames": 100,
            "window_duration_sec": boundary + len(ch0_gen) / sr,
            "rtf": None,
            "channels": [
                {
                    "gen_wav": f"wav/{wid}_ch0.wav",
                    "prompt_wav": f"wav/{wid}_ch0.wav",
                    "gt_wav": f"wav/{wid}_ch0.wav",
                    "ref_text": "",
                },
                {
                    "gen_wav": f"wav/{wid}_ch1.wav",
                    "prompt_wav": f"wav/{wid}_ch1.wav",
                    "gt_wav": f"wav/{wid}_ch1.wav",
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

    def test_writes_jsonl_and_summary_with_the_documented_keys(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        ch0 = _block(0.5, 0.11, sr)
        ch1 = _block(0.5, 0.22, sr)
        self._write_window(test_dir, "sess_w00000", ch0, ch1)

        backend = KeyedFakeMOSBackend()
        backend.register(ch0, 3.0)
        backend.register(ch1, 4.0)

        metric = ChannelQualityMetric(mos_backend=backend, vad=_FrameEnergyVAD())
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert set(summary) == {"utmos_mean"}
        assert summary["utmos_mean"] == pytest.approx(3.5)  # mean of ch0/ch1

        scoring_dir = inference_dir / "valid" / "scoring" / "channel_quality"
        lines = (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["window_id"] == "sess_w00000"
        assert record["utmos_window_mean"] == pytest.approx(3.5)
        assert "dnsmos" not in record["channels"][0]

        on_disk_summary = json.loads((scoring_dir / "summary.json").read_text("utf-8"))
        assert on_disk_summary == summary

    def test_dnsmos_key_appears_only_when_enabled(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        ch0 = _block(0.5, 0.11, sr)
        ch1 = _block(0.5, 0.22, sr)
        self._write_window(test_dir, "sess_w00000", ch0, ch1)

        mos_backend = KeyedFakeMOSBackend()
        mos_backend.register(ch0, 3.0)
        mos_backend.register(ch1, 4.0)

        class _FakeDnsmos:
            def __call__(self, wav, sr):
                return float(np.max(np.abs(wav)) * 10.0)

        metric = ChannelQualityMetric(
            mos_backend=mos_backend,
            vad=_FrameEnergyVAD(),
            enable_dnsmos=True,
            dnsmos_backend=_FakeDnsmos(),
        )
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert set(summary) == {"utmos_mean", "dnsmos_mean"}
        # ch0 amp=0.11 -> dnsmos 1.1, ch1 amp=0.22 -> dnsmos 2.2, mean 1.65
        assert summary["dnsmos_mean"] == pytest.approx(1.65, abs=1e-6)

    def test_meta_relative_paths_resolve_against_the_test_dir(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        ch0 = _block(0.5, 0.5, sr)
        ch1 = np.zeros(int(0.3 * sr), dtype=np.float32)
        self._write_window(test_dir, "sess_w00000", ch0, ch1)

        backend = KeyedFakeMOSBackend()
        backend.register(ch0, 4.5)

        metric = ChannelQualityMetric(mos_backend=backend, vad=_FrameEnergyVAD())
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)
        # ch1 is silent -> no IPUs -> None, excluded; window mean is ch0
        # alone -> 4.5.
        assert summary["utmos_mean"] == pytest.approx(4.5)


# --------------------------------------------------------------------------- #
# conf/metrics.yaml wiring: offline instantiation
# --------------------------------------------------------------------------- #
class TestMetricsConfigInstantiatesOffline:
    def test_channel_quality_metric_entry_instantiates_without_network(
        self, monkeypatch
    ):
        import torch.hub
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

        def guard(*args, **kwargs):
            raise AssertionError("torch.hub.load called while instantiating config")

        monkeypatch.setattr(torch.hub, "load", guard)

        entries = [
            entry
            for entry in metrics_config.metrics
            if entry.metric._target_.endswith("ChannelQualityMetric")
        ]
        assert len(entries) == 1
        metric = instantiate(entries[0].metric)
        assert isinstance(metric, ChannelQualityMetric)
        assert metric.mos_backend._predictor is None
        assert metric.enable_dnsmos is False
