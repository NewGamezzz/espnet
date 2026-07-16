"""``QualityMetric`` (``src/metrics/quality.py``) tests.

Fake MOS backend, CPU-only, no network: covers exactly ONE predictor call
per window on the ``mix_wav`` (never per-channel, never per-IPU), the plain
mean over windows, backend laziness for the default UTMOS backend
(``torch.hub``-based, see the module docstring's documented deviation from
PLAN-step4.md's "speechmos" wording), and a full ``__call__`` round trip
against a fabricated ``inference_dir``.

The real ``torch.hub`` UTMOS backend is never exercised end to end here --
see ``TorchHubUTMOSBackend``'s docstring for why no live network smoke test
is shipped (the binding "no GPU/model downloads locally" constraint, and no
import-based flag exists to gate a ``torch.hub`` fetch the way
``_HAS_TRANSFORMERS`` gates ``transformers``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.quality import (
    QualityMetric,
    TorchHubUTMOSBackend,
)


# --------------------------------------------------------------------------- #
# test-only fakes
# --------------------------------------------------------------------------- #
class KeyedFakeMOSBackend:
    """Deterministic fake MOS backend: looks up a score by the EXACT sample
    content it's called with, and counts its own calls (so a test can prove
    "one call per window", not just "the right score came back")."""

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


def _write_wav_exact(path: Path, data: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sr, subtype="FLOAT")


def _block(duration_s: float, amplitude: float, sr: int) -> np.ndarray:
    return np.full(int(round(duration_s * sr)), amplitude, dtype=np.float32)


def _write_window(test_dir: Path, wid: str, mix_wav: np.ndarray, sr: int) -> None:
    _write_wav_exact(test_dir / "mix" / f"{wid}.wav", mix_wav, sr)
    meta = {
        "window_id": wid,
        "session_id": "sess",
        "mode": "generate",
        "sample_rate": sr,
        "num_channels": 2,
        "window_duration_sec": 12.0,
        "rtf": None,
        "mix_wav": f"mix/{wid}.wav",
        "prompt": {"total_sec": 4.0, "total_frames": 375, "turns": []},
        "channels": [],
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

    def test_metric_construction_with_all_real_defaults_does_not_touch_network(self):
        metric = QualityMetric()
        assert isinstance(metric.mos_backend, TorchHubUTMOSBackend)
        assert metric.mos_backend._predictor is None


# --------------------------------------------------------------------------- #
# full __call__ round trip
# --------------------------------------------------------------------------- #
class TestCallRoundTrip:
    SR = 16000

    def test_one_mos_call_per_window_on_the_mix_path(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0 = _block(0.5, 0.11, sr)
        _write_window(test_dir, "sess_w00000", mix0, sr)
        _write_meta_scp(test_dir, ["sess_w00000"])

        backend = KeyedFakeMOSBackend()
        backend.register(mix0, 4.0)

        metric = QualityMetric(mos_backend=backend)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert len(backend.calls) == 1  # exactly one call, on the mixdown
        assert summary["utmos_mean"] == pytest.approx(4.0)

    def test_mean_over_windows_not_a_fabricated_weighting(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0 = _block(0.5, 0.11, sr)
        mix1 = _block(2.0, 0.22, sr)  # different duration: proves no
        # duration-weighting sneaks back in (plain mean of window scores).
        _write_window(test_dir, "sess_w00000", mix0, sr)
        _write_window(test_dir, "sess_w00001", mix1, sr)
        _write_meta_scp(test_dir, ["sess_w00000", "sess_w00001"])

        backend = KeyedFakeMOSBackend()
        backend.register(mix0, 2.0)
        backend.register(mix1, 4.0)

        metric = QualityMetric(mos_backend=backend)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert len(backend.calls) == 2
        assert summary["utmos_mean"] == pytest.approx(3.0)  # plain (2+4)/2

    def test_writes_jsonl_and_summary_with_the_documented_keys(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0 = _block(0.5, 0.11, sr)
        _write_window(test_dir, "sess_w00000", mix0, sr)
        _write_meta_scp(test_dir, ["sess_w00000"])

        backend = KeyedFakeMOSBackend()
        backend.register(mix0, 3.5)

        metric = QualityMetric(mos_backend=backend)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert set(summary) == {"utmos_mean"}
        scoring_dir = inference_dir / "valid" / "scoring" / "quality"
        lines = (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["window_id"] == "sess_w00000"
        assert record["score"] == pytest.approx(3.5)

        on_disk_summary = json.loads((scoring_dir / "summary.json").read_text("utf-8"))
        assert on_disk_summary == summary

    def test_meta_relative_paths_resolve_against_the_test_dir(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0 = _block(0.5, 0.5, sr)
        _write_window(test_dir, "sess_w00000", mix0, sr)
        _write_meta_scp(test_dir, ["sess_w00000"])

        backend = KeyedFakeMOSBackend()
        backend.register(mix0, 4.5)

        metric = QualityMetric(mos_backend=backend)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)
        assert summary["utmos_mean"] == pytest.approx(4.5)

    def test_resamples_native_rate_mix_to_the_backends_target_rate(self, tmp_path):
        # Recipe audio is 24 kHz; the backend must see it resampled to
        # quality_sample_rate (16 kHz default) -- registering the fake at
        # 16 kHz and writing the fixture at 24 kHz proves the resample path,
        # not just that SOME array reaches the backend.
        native_sr = 24000
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        mix0_native = _block(0.5, 0.3, native_sr)
        _write_window(test_dir, "sess_w00000", mix0_native, native_sr)
        _write_meta_scp(test_dir, ["sess_w00000"])

        calls = []

        def _always_score(wav, sr):
            calls.append(np.asarray(wav))
            assert sr == 16000
            return 3.0

        metric = QualityMetric(mos_backend=_always_score, quality_sample_rate=16000)
        summary = metric({"meta": test_dir / "meta.scp"}, "valid", inference_dir)

        assert len(calls) == 1
        assert summary["utmos_mean"] == pytest.approx(3.0)


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
