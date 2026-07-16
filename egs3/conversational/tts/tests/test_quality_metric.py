"""``QualityMetric`` (``src/metrics/quality.py``) tests.

Fake MOS + VAD backends, CPU-only, no network: covers the per-IPU scoring
path (VAD spans cropped from each channel's wav, short-IPU filtering, the
POOLED per-IPU mean rather than a mean of window means), the whole-mixdown
score per window, the three documented summary keys, backend laziness for
both real defaults, and a full ``__call__`` round trip against a fabricated
``inference_dir``.

The real ``torch.hub`` UTMOS backend is never exercised end to end here --
see ``TorchHubUTMOSBackend``'s docstring for why no live network smoke test
is shipped. The real Silero VAD backend gets one import-gated smoke test
(``faster-whisper`` ships the ONNX model in its wheel, so when the package
is installed the test runs offline).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.quality import (
    QualityMetric,
    SileroVADSegmenter,
    TorchHubUTMOSBackend,
)

try:  # pragma: no cover - environment probe
    import faster_whisper  # noqa: F401

    _HAS_FASTER_WHISPER = True
except ImportError:  # pragma: no cover
    _HAS_FASTER_WHISPER = False


# --------------------------------------------------------------------------- #
# test-only fakes
# --------------------------------------------------------------------------- #
class KeyedFakeMOSBackend:
    """Deterministic fake MOS backend: looks up a score by the EXACT sample
    content it's called with, and counts its own calls (so a test can prove
    "one call per unit", not just "the right score came back")."""

    def __init__(self):
        self._table: dict[tuple, float] = {}
        self.calls: list[np.ndarray] = []

    @staticmethod
    def _key(wav) -> tuple:
        return tuple(np.round(np.asarray(wav, dtype=np.float64), 6).tolist())

    def register(self, wav: np.ndarray, score: float) -> np.ndarray:
        arr = np.asarray(wav, dtype=np.float32)
        self._table[self._key(arr)] = float(score)
        return arr

    def __call__(self, wav, sr):
        self.calls.append(np.asarray(wav))
        key = self._key(wav)
        if key not in self._table:
            raise KeyError(
                f"no fake MOS score registered for snippet of len {len(wav)}"
            )
        return self._table[key]


class KeyedFakeVADBackend:
    """Deterministic fake VAD: returns registered spans (seconds) for the
    EXACT wav it's called with; unknown wavs get no speech ([])."""

    def __init__(self):
        self._table: dict[tuple, list] = {}
        self.calls: list[np.ndarray] = []

    def register(self, wav: np.ndarray, spans) -> np.ndarray:
        arr = np.asarray(wav, dtype=np.float32)
        self._table[KeyedFakeMOSBackend._key(arr)] = list(spans)
        return arr

    def __call__(self, wav, sr):
        self.calls.append(np.asarray(wav))
        return self._table.get(KeyedFakeMOSBackend._key(wav), [])


def _write_wav_exact(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def _block(duration_s: float, amplitude: float, sr: int) -> np.ndarray:
    return np.full(int(round(duration_s * sr)), amplitude, dtype=np.float32)


def _write_window(
    test_dir: Path,
    wid: str,
    mix_wav: np.ndarray,
    sr: int,
    channel_wavs: list | None = None,
) -> None:
    _write_wav_exact(test_dir / "mix" / f"{wid}.wav", mix_wav, sr)
    channels = []
    for ch, wav in enumerate(channel_wavs or []):
        rel = f"wav/{wid}_ch{ch}.wav"
        _write_wav_exact(test_dir / rel, wav, sr)
        channels.append(
            {
                "gen_wav": rel,
                "prompt_wav": rel,
                "gt_wav": rel,
                "ref_text": "",
            }
        )
    meta = {
        "window_id": wid,
        "session_id": "sess",
        "mode": "generate",
        "sample_rate": sr,
        "num_channels": max(len(channels), 2),
        "window_duration_sec": 12.0,
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
    def test_torch_hub_backend_construction_does_not_call_torch_hub_load(
        self, monkeypatch
    ):
        import torch.hub

        def guard(*args, **kwargs):
            raise AssertionError("torch.hub.load called before first __call__")

        monkeypatch.setattr(torch.hub, "load", guard)
        backend = TorchHubUTMOSBackend()
        assert backend._predictor is None

    def test_vad_segmenter_construction_does_not_import_faster_whisper(self):
        segmenter = SileroVADSegmenter()
        assert segmenter._get_speech_timestamps is None

    def test_metric_construction_with_all_real_defaults_does_not_touch_network(self):
        metric = QualityMetric()
        assert isinstance(metric.mos_backend, TorchHubUTMOSBackend)
        assert metric.mos_backend._predictor is None
        assert isinstance(metric.vad_backend, SileroVADSegmenter)
        assert metric.vad_backend._get_speech_timestamps is None


# --------------------------------------------------------------------------- #
# VAD segmenter contract
# --------------------------------------------------------------------------- #
class TestSileroVADSegmenter:
    def test_rejects_non_16k_audio_before_any_import(self):
        segmenter = SileroVADSegmenter()
        with pytest.raises(ValueError, match="16000"):
            segmenter(np.zeros(24000, dtype=np.float32), 24000)
        assert segmenter._get_speech_timestamps is None  # sr check comes first

    @pytest.mark.skipif(
        not _HAS_FASTER_WHISPER, reason="faster-whisper not installed"
    )
    def test_real_silero_finds_speech_not_silence(self):
        # 1 s silence + 2 s noise burst + 1 s silence: exactly one span,
        # roughly over the burst. The ONNX model ships in the wheel, so this
        # runs offline.
        sr = 16000
        rng = np.random.default_rng(0)
        wav = np.concatenate(
            [
                np.zeros(sr, dtype=np.float32),
                (rng.standard_normal(2 * sr) * 0.3).astype(np.float32),
                np.zeros(sr, dtype=np.float32),
            ]
        )
        spans = SileroVADSegmenter()(wav, sr)
        assert len(spans) >= 1
        start, end = spans[0]
        assert 0.5 <= start <= 1.5
        assert end - start >= 1.0


# --------------------------------------------------------------------------- #
# full __call__ round trip
# --------------------------------------------------------------------------- #
class TestCallRoundTrip:
    SR = 16000

    def test_mix_score_plus_per_ipu_scores_with_short_filtering(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"

        mix0 = _block(0.5, 0.11, sr)
        ch0 = np.concatenate([_block(2.0, 0.2, sr), _block(2.0, 0.0, sr)])
        ch1 = np.concatenate([_block(1.0, 0.0, sr), _block(3.0, 0.4, sr)])
        _write_window(test_dir, "sess_w00000", mix0, sr, [ch0, ch1])
        _write_meta_scp(test_dir, ["sess_w00000"])

        vad = KeyedFakeVADBackend()
        # ch0: one scoreable IPU and one below min_ipu_sec (filtered, counted)
        vad.register(ch0, [(0.0, 2.0), (2.5, 3.0)])
        vad.register(ch1, [(1.0, 4.0)])

        mos = KeyedFakeMOSBackend()
        mos.register(mix0, 2.0)
        mos.register(ch0[0 : 2 * sr], 4.0)
        mos.register(ch1[1 * sr : 4 * sr], 3.0)

        metric = QualityMetric(mos_backend=mos, vad_backend=vad, min_ipu_sec=1.0)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        # one mix call + two IPU calls; the 0.5 s span never reaches MOS
        assert len(mos.calls) == 3
        assert summary["utmos_ipu_mean"] == pytest.approx(3.5)  # (4+3)/2
        assert summary["utmos_mix_mean"] == pytest.approx(2.0)
        assert summary["ipu_count"] == pytest.approx(2.0)

        record = json.loads(
            (inference_dir / "valid" / "scoring" / "quality" / "windows.jsonl")
            .read_text("utf-8")
            .splitlines()[0]
        )
        assert record["window_id"] == "sess_w00000"
        assert record["mix_score"] == pytest.approx(2.0)
        assert record["n_ipus_skipped_short"] == 1
        assert [(i["channel"], i["start"], i["end"]) for i in record["ipus"]] == [
            (0, 0.0, 2.0),
            (1, 1.0, 4.0),
        ]

    def test_ipu_mean_pools_over_ipus_not_over_window_means(self, tmp_path):
        # Window A: one IPU scored 1.0; window B: three IPUs scored 3.0 each.
        # Pooled mean = (1 + 3*3)/4 = 2.5; a mean of window means would be 2.0.
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"

        mix_a = _block(0.5, 0.11, sr)
        mix_b = _block(0.5, 0.22, sr)
        ch_a = _block(6.0, 0.2, sr)
        ch_b = _block(6.0, 0.4, sr)
        _write_window(test_dir, "sess_w00000", mix_a, sr, [ch_a])
        _write_window(test_dir, "sess_w00001", mix_b, sr, [ch_b])
        _write_meta_scp(test_dir, ["sess_w00000", "sess_w00001"])

        vad = KeyedFakeVADBackend()
        vad.register(ch_a, [(0.0, 1.0)])
        vad.register(ch_b, [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)])

        mos = KeyedFakeMOSBackend()
        mos.register(mix_a, 5.0)
        mos.register(mix_b, 5.0)
        mos.register(ch_a[0:sr], 1.0)
        mos.register(ch_b[0:sr], 3.0)
        mos.register(ch_b[2 * sr : 3 * sr], 3.0)
        mos.register(ch_b[4 * sr : 5 * sr], 3.0)

        metric = QualityMetric(mos_backend=mos, vad_backend=vad)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert summary["utmos_ipu_mean"] == pytest.approx(2.5)  # pooled
        assert summary["ipu_count"] == pytest.approx(4.0)

    def test_no_channels_yields_null_ipu_mean_and_zero_count(self, tmp_path):
        # A window with no channel wavs (or a VAD that finds no speech)
        # leaves utmos_ipu_mean undefined -> None, never a fabricated 0.0;
        # ipu_count 0.0 is an honest count, not an undefined value.
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0 = _block(0.5, 0.11, sr)
        _write_window(test_dir, "sess_w00000", mix0, sr)
        _write_meta_scp(test_dir, ["sess_w00000"])

        mos = KeyedFakeMOSBackend()
        mos.register(mix0, 4.0)

        metric = QualityMetric(mos_backend=mos, vad_backend=KeyedFakeVADBackend())
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert summary["utmos_ipu_mean"] is None
        assert summary["utmos_mix_mean"] == pytest.approx(4.0)
        assert summary["ipu_count"] == pytest.approx(0.0)

    def test_writes_jsonl_and_summary_with_the_documented_keys(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0 = _block(0.5, 0.11, sr)
        _write_window(test_dir, "sess_w00000", mix0, sr)
        _write_meta_scp(test_dir, ["sess_w00000"])

        mos = KeyedFakeMOSBackend()
        mos.register(mix0, 3.5)

        metric = QualityMetric(mos_backend=mos, vad_backend=KeyedFakeVADBackend())
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert set(summary) == {"utmos_ipu_mean", "utmos_mix_mean", "ipu_count"}
        scoring_dir = inference_dir / "valid" / "scoring" / "quality"
        on_disk_summary = json.loads((scoring_dir / "summary.json").read_text("utf-8"))
        assert on_disk_summary == summary

    def test_resamples_native_rate_audio_to_the_backends_target_rate(self, tmp_path):
        # Recipe audio is 24 kHz; both backends must see it resampled to
        # quality_sample_rate (16 kHz default).
        native_sr = 24000
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0_native = _block(0.5, 0.3, native_sr)
        ch0_native = _block(2.0, 0.2, native_sr)
        _write_window(test_dir, "sess_w00000", mix0_native, native_sr, [ch0_native])
        _write_meta_scp(test_dir, ["sess_w00000"])

        seen_srs = []

        def _score_anything(wav, sr):
            seen_srs.append(sr)
            return 3.0

        def _vad_anything(wav, sr):
            seen_srs.append(sr)
            return [(0.0, 1.0)]

        metric = QualityMetric(
            mos_backend=_score_anything,
            vad_backend=_vad_anything,
            quality_sample_rate=16000,
        )
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert seen_srs and all(sr == 16000 for sr in seen_srs)
        assert summary["utmos_mix_mean"] == pytest.approx(3.0)
        assert summary["utmos_ipu_mean"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# conf/metrics.yaml wiring: offline instantiation
# --------------------------------------------------------------------------- #
class TestMetricsConfigInstantiatesOffline:
    def test_quality_metric_entry_instantiates_without_network(self, monkeypatch):
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
            if entry.metric._target_.endswith("QualityMetric")
        ]
        assert len(entries) == 1
        metric = instantiate(entries[0].metric)
        assert isinstance(metric, QualityMetric)
        assert metric.mos_backend._predictor is None
        assert metric.vad_backend._get_speech_timestamps is None
