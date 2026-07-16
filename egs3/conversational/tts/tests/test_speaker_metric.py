"""``SpeakerSimilarityMetric`` (``src/metrics/speaker.py``) tests.

Fake-embedder, CPU-only, no network: covers the pure ``_cosine`` helper, the
``embed_min_sec`` minimal-duration guard (skip-and-count, never a fabricated
0.0), backend laziness, and a full ``__call__`` round trip against a
fabricated ``inference_dir`` matching ``src/inference.py``'s current meta
contract (``channels[ch].prompt_wav`` / ``channels[ch].gen_wav``, whole
files, no VAD/IPU segmentation).

Real WavLM-SV (``transformers``) is only exercised by the asset-gated smoke
test at the bottom.

The fake embedder used throughout (``KeyedFakeEmbedder``) keys on the EXACT
sample content of the array it's called with (not a summary statistic like
its mean). Test wavs are written at 16 kHz (the embedder's fixed native
rate, and this metric's default ``embed_sample_rate``) with
``subtype="FLOAT"`` so the float32 samples round-trip through disk
bit-exactly and the metric's post-resample array matches the registered key
(a non-16 kHz fixture would otherwise resample before the embedder ever
sees it, breaking the exact-content key).
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.metrics.speaker import (
    SpeakerSimilarityMetric,
    WavLMSVEmbedder,
    _cosine,
)

try:
    import transformers  # noqa: F401

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# --------------------------------------------------------------------------- #
# test-only fakes
# --------------------------------------------------------------------------- #
class KeyedFakeEmbedder:
    """Deterministic fake embedder: looks up a vector by the EXACT sample
    content it's called with."""

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
# _cosine: pure math helper
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
        metric = SpeakerSimilarityMetric()
        assert isinstance(metric.embedder, WavLMSVEmbedder)
        assert metric.embedder.model_tag == "microsoft/wavlm-base-plus-sv"

    def test_embedder_rejects_non_16k_audio_without_loading_the_model(self):
        embedder = WavLMSVEmbedder()
        with pytest.raises(ValueError, match="16000"):
            embedder(np.zeros(100, dtype=np.float32), 8000)
        assert embedder._model is None


# --------------------------------------------------------------------------- #
# full __call__ round trip: cosine per channel, the embed_min_sec floor
# (skip-and-count), and the documented summary-key set.
# --------------------------------------------------------------------------- #
class TestCallRoundTrip:
    SR = 16000

    def _write_window(
        self,
        test_dir: Path,
        wid: str,
        gen_wavs: dict,
        prompt_wavs: dict,
    ) -> None:
        sr = self.SR
        channels = []
        for ch in sorted(gen_wavs):
            _write_wav_exact(test_dir / "wav" / f"{wid}_ch{ch}.wav", gen_wavs[ch], sr)
            _write_wav_exact(
                test_dir / "prompt" / f"{wid}_ch{ch}.wav", prompt_wavs[ch], sr
            )
            channels.append(
                {
                    "gen_wav": f"wav/{wid}_ch{ch}.wav",
                    "prompt_wav": f"prompt/{wid}_ch{ch}.wav",
                    "gt_wav": f"wav/{wid}_ch{ch}.wav",
                    "ref_text": "",
                }
            )
        _write_wav_exact(
            test_dir / "mix" / f"{wid}.wav", next(iter(gen_wavs.values())), sr
        )
        meta = {
            "window_id": wid,
            "session_id": "sess",
            "mode": "generate",
            "sample_rate": sr,
            "num_channels": len(gen_wavs),
            "window_duration_sec": 12.0,
            "rtf": None,
            "mix_wav": f"mix/{wid}.wav",
            "prompt": {"total_sec": 4.0, "total_frames": 375, "turns": []},
            "channels": channels,
            "turns": [],
        }
        (test_dir / "meta").mkdir(parents=True, exist_ok=True)
        (test_dir / f"meta/{wid}.json").write_text(json.dumps(meta), encoding="utf-8")
        (test_dir / "meta.scp").write_text(f"{wid} meta/{wid}.json\n", encoding="utf-8")

    def test_cosine_per_channel_with_deterministic_embedder(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"

        gen0, prompt0 = _block(0.5, 0.4, sr), _block(0.5, 0.4, sr)  # identical
        gen1, prompt1 = _block(0.5, 0.4, sr), _block(0.5, -0.4, sr)  # opposite
        self._write_window(
            test_dir, "sess_w00000", {0: gen0, 1: gen1}, {0: prompt0, 1: prompt1}
        )

        embedder = KeyedFakeEmbedder()
        embedder.register(gen0, [1.0, 0.0])
        embedder.register(prompt0, [1.0, 0.0])
        embedder.register(gen1, [1.0, 0.0])
        embedder.register(prompt1, [-1.0, 0.0])

        metric = SpeakerSimilarityMetric(embedder=embedder, embed_min_sec=0.1)
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_similarity"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert record["channels"][0]["cosine"] == pytest.approx(1.0)
        assert record["channels"][1]["cosine"] == pytest.approx(-1.0)
        assert record["n_skipped"] == 0
        # flat mean over the two channel cosines: (1.0 + -1.0) / 2 = 0.0.
        assert summary["sim_o_mean"] == pytest.approx(0.0)

    def test_short_audio_below_floor_yields_none_and_is_skip_counted(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"

        gen0 = _block(0.5, 0.4, sr)
        prompt0 = _block(0.05, 0.4, sr)  # below the 0.1s floor used here
        self._write_window(test_dir, "sess_w00000", {0: gen0}, {0: prompt0})

        embedder = KeyedFakeEmbedder()
        embedder.register(gen0, [1.0, 0.0])
        # prompt0 deliberately NOT registered: if the metric embedded it
        # anyway (floor not enforced), the fake would raise KeyError.

        metric = SpeakerSimilarityMetric(embedder=embedder, embed_min_sec=0.1)
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_similarity"
        record = json.loads(
            (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()[0]
        )
        assert record["channels"][0]["cosine"] is None
        assert record["channels"][0]["skipped"] is True
        assert record["n_skipped"] == 1
        assert summary["sim_o_mean"] is None

    def test_summary_is_none_when_every_channel_is_skipped(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"

        gen0 = _block(0.05, 0.4, sr)  # both sides below the floor
        prompt0 = _block(0.05, 0.4, sr)
        self._write_window(test_dir, "sess_w00000", {0: gen0}, {0: prompt0})

        metric = SpeakerSimilarityMetric(
            embedder=KeyedFakeEmbedder(), embed_min_sec=0.1
        )
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        assert set(summary) == {"sim_o_mean"}
        assert summary["sim_o_mean"] is None

        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_similarity"
        summary_text = (scoring_dir / "summary.json").read_text("utf-8")
        # Reaches disk as JSON null, which local/eval_report.py's
        # _format_cell renders as "-".
        assert '"sim_o_mean": null' in summary_text

    def test_writes_jsonl_and_summary_with_the_documented_keys(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        gen0, prompt0 = _block(0.5, 0.2, sr), _block(0.5, 0.2, sr)
        self._write_window(test_dir, "sess_w00000", {0: gen0}, {0: prompt0})

        embedder = KeyedFakeEmbedder()
        embedder.register(gen0, [1.0, 0.0])
        embedder.register(prompt0, [1.0, 0.0])

        metric = SpeakerSimilarityMetric(embedder=embedder, embed_min_sec=0.1)
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)

        assert set(summary) == {"sim_o_mean"}
        scoring_dir = inference_dir / "valid" / "scoring" / "speaker_similarity"
        lines = (scoring_dir / "windows.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["window_id"] == "sess_w00000"

        on_disk_summary = json.loads((scoring_dir / "summary.json").read_text("utf-8"))
        assert on_disk_summary == summary

    def test_meta_relative_paths_resolve_against_the_test_dir(self, tmp_path):
        sr = self.SR
        inference_dir = tmp_path / "infer"
        test_dir = inference_dir / "valid"
        gen0, prompt0 = _block(0.5, 0.4, sr), _block(0.5, 0.4, sr)
        self._write_window(test_dir, "sess_w00000", {0: gen0}, {0: prompt0})

        embedder = KeyedFakeEmbedder()
        embedder.register(gen0, [1.0, 0.0])
        embedder.register(prompt0, [1.0, 0.0])

        metric = SpeakerSimilarityMetric(embedder=embedder, embed_min_sec=0.1)
        data = {"meta": test_dir / "meta.scp"}
        summary = metric(data, "valid", inference_dir)
        assert summary["sim_o_mean"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# conf/metrics.yaml wiring: the binding constraint that the shipped config
# instantiates every metric offline with its real (lazy) defaults.
# --------------------------------------------------------------------------- #
class TestMetricsConfigInstantiatesOffline:
    def test_speaker_similarity_metric_entry_instantiates_without_network(
        self, monkeypatch
    ):
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
            if entry.metric._target_.endswith("SpeakerSimilarityMetric")
        ]
        assert len(speaker_entries) == 1
        metric = instantiate(speaker_entries[0].metric)
        assert isinstance(metric, SpeakerSimilarityMetric)
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
        assert len(metrics) == 3
        assert any(isinstance(m, SpeakerSimilarityMetric) for m in metrics)


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
