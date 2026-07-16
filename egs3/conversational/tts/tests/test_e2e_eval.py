"""End-to-end smoke: ``infer`` -> ``measure`` through the stage entry points.

Drives ``ConversationalTTSSystem.infer()`` / ``run_inference`` then
``ConversationalTTSSystem.measure()`` / ``measure()`` on a fully fabricated
fixture (a tiny two-channel FLAC, a hand-built window manifest, and the tiny
random-init DiT from the trainer suite), proving the step's acceptance
criterion end to end: ``meta.scp`` -> all three lean metric classes -> a
``metrics.json`` with every documented summary key present. No corpus, no
checkpoint, no network: every metric backend (transcriber, normalizer,
embedder, MOS predictor) is swapped for a trivial fake, injected through a
test-scoped metrics config that is hydra-instantiated the same way
``conf/metrics.yaml`` is (mirroring ``tests/test_measure.py``'s stub-metric
approach, extended here to the three REAL metric classes).

Two runs exercise the two viable ``infer`` code paths, BOTH on a two-window
session: under the reworked infer stage's leakage rule (a channel's prompt
turn may never come from inside the evaluated window), a session with only
one window has an empty prompt candidate pool for every channel and every
window is skipped, so both fixtures below give the session a second window
whose turns serve as the other window's prompt candidates -- the same
two-window-per-session pattern ``tests/test_inference.py`` established.

* ``gt`` mode, fully through ``ConversationalTTSSystem`` (``system.infer()``
  then ``system.measure()``) -- the literal ``python run.py --stages infer``
  / ``--stages measure`` dispatch, and the cheapest mode (no model, no
  vocoder needed).
* ``generate`` mode, through ``run_inference`` directly with the tiny DiT +
  a fake Vocos injected (the same seam ``tests/test_inference.py`` uses for
  its generate-mode tests) -- proves the plumbing also holds for real model
  sampling.

Per-metric correctness (pooled WER, SIM-o, UTMOS weighting, ...) is the job
of ``tests/test_asr_metric.py`` / ``test_speaker_metric.py`` /
``test_quality_metric.py``; this file only proves the plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from .conftest import EXT_TOKENS
from .test_build_model import build_tiny  # noqa: F401  (fixture reuse)
from .test_inference import FakeVocoder, _infer_config, _write_flac

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.windows import (
    WindowRecord,
    to_json,
)
from egs3.conversational.tts.src.inference import run_inference
from egs3.conversational.tts.src.system import ConversationalTTSSystem
from espnet3.systems.base.metric import measure

FS = 24000
SRC_SR = 48000
HOP = 256

ASR_SUMMARY_KEYS = {"wer_channel", "wer_mix"}
SPEAKER_SUMMARY_KEYS = {"sim_o_mean"}
QUALITY_SUMMARY_KEYS = {"utmos_ipu_mean", "utmos_mix_mean", "ipu_count"}
INTERACTION_SUMMARY_KEYS = {
    f"{event}_{suffix}"
    for event in ("ipu", "pause", "gap", "overlap")
    for suffix in ("per_min", "sec_per_min", "dur_w1")
}


# --------------------------------------------------------------------------- #
# trivial fake backends, hydra-instantiated by the test-scoped metrics config
# below (``_target_: <this module>.<ClassName>``) exactly like
# ``tests/test_measure.py``'s ``StubMetaMetric`` is targeted by ``__name__``.
# --------------------------------------------------------------------------- #
class FakeTranscriber:
    """One fixed hypothesis regardless of input; proves the ASR plumbing
    wires up (WER bookkeeping itself is ``tests/test_asr_metric.py``'s
    job)."""

    def __call__(self, wav, sr):
        if len(wav) == 0:
            return ""
        return "ok"


class FakeNormalizer:
    def __call__(self, text):
        return text.lower()


class FakeEmbedder:
    """Fixed unit vector: any two calls cosine to 1.0, so ``sim_o_mean`` is
    always defined (never a None-only fallback path)."""

    def __call__(self, wav, sr):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class FakeMOSBackend:
    def __call__(self, wav, sr):
        return 3.5


class FakeVADBackend:
    """Whole wav as one speech span: every channel yields exactly one IPU
    (VAD correctness itself is ``tests/test_quality_metric.py``'s job)."""

    def __call__(self, wav, sr):
        if len(wav) == 0:
            return []
        return [(0.0, len(wav) / sr)]


def _fake(cls_name: str) -> dict:
    return {"_target_": f"{__name__}.{cls_name}"}


def _fake_metrics_config(inference_dir: Path) -> OmegaConf:
    """The three real metric classes, hydra-instantiated with fake backends
    injected via nested ``_target_`` entries (recursive instantiate is
    already the house pattern, e.g. ``conf/training_poc.yaml``'s nested
    ``preprocessor:`` block)."""
    return OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": "valid"}]},
            "metrics": [
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics.asr."
                            "ConversationASRMetric"
                        ),
                        "transcriber": _fake("FakeTranscriber"),
                        "normalizer": _fake("FakeNormalizer"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics.speaker."
                            "SpeakerSimilarityMetric"
                        ),
                        "embedder": _fake("FakeEmbedder"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics.quality."
                            "QualityMetric"
                        ),
                        "mos_backend": _fake("FakeMOSBackend"),
                        "vad_backend": _fake("FakeVADBackend"),
                    },
                    "inputs": {"meta": "meta"},
                },
                {
                    "metric": {
                        "_target_": (
                            "egs3.conversational.tts.src.metrics.interaction."
                            "InteractionMetric"
                        ),
                        "vad_backend": _fake("FakeVADBackend"),
                    },
                    "inputs": {"meta": "meta"},
                },
            ],
        }
    )


def _assert_all_summary_keys_present(results: dict) -> None:
    for suffix, expected in (
        ("ConversationASRMetric", ASR_SUMMARY_KEYS),
        ("SpeakerSimilarityMetric", SPEAKER_SUMMARY_KEYS),
        ("QualityMetric", QUALITY_SUMMARY_KEYS),
        ("InteractionMetric", INTERACTION_SUMMARY_KEYS),
    ):
        matches = [k for k in results if k.endswith(suffix)]
        assert len(matches) == 1, f"expected exactly one {suffix} entry, got {matches}"
        summary = results[matches[0]]["valid"]
        missing = expected - set(summary)
        assert not missing, f"{suffix} summary missing keys: {missing}"
        # A key can legitimately be undefined (None, not a fabricated
        # number) when this fixture's fakes never produced data for it --
        # see src/metrics/_common.py's summary_value.
        assert all(isinstance(v, float) or v is None for v in summary.values())


# --------------------------------------------------------------------------- #
# fixture: two windows on one FLAC (feeds the gt-mode-via-system run below);
# a single-window variant (matching tests/test_inference.py's fixture shape)
# feeds the generate-mode leg.
# --------------------------------------------------------------------------- #
def _turns_at(offset: float) -> tuple:
    return (
        Turn(0, "spk_a", "abc def", offset + 0.5, offset + 3.0),
        Turn(1, "spk_b", "bead cab", offset + 3.4, offset + 5.5),
        Turn(0, "spk_a", "fade dad", offset + 6.0, offset + 8.5),
        Turn(1, "spk_b", "bag", offset + 9.0, offset + 10.0),
    )


def _window(window_id: str, t0: float) -> WindowRecord:
    return WindowRecord(
        window_id=window_id,
        session_id="sess",
        audio_relpath="original/sess_mixed.flac",
        num_channels=2,
        sample_rate=SRC_SR,
        t0=t0,
        t1=t0 + 12.0,
        turns=_turns_at(t0),
    )


def _build_fixture(tmp_path, windows, flac_duration_s: float) -> dict:
    """One fabricated recipe data dir (FLAC + manifest + vocab + minimal
    training config); the two fixtures below vary only the window list and
    the FLAC duration that must cover it."""
    root = tmp_path / "data"
    _write_flac(root / "original" / "sess_mixed.flac", 2, flac_duration_s, SRC_SR)
    manifest = root / "valid.jsonl"
    manifest.write_text(
        "".join(json.dumps(to_json(w)) + "\n" for w in windows), encoding="utf-8"
    )
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(EXT_TOKENS) + "\n", encoding="utf-8")
    training_config = OmegaConf.create(
        {
            "recipe_dir": str(tmp_path),
            "sample_rate": FS,
            "hop_length": HOP,
            "dataset": {"preprocessor": {"token_list": str(vocab)}},
        }
    )
    return {
        "tmp_path": tmp_path,
        "manifest": manifest,
        "dataset_root": root,
        "vocab": vocab,
        "training_config": training_config,
    }


@pytest.fixture
def two_window_fixture(tmp_path):
    windows = [_window("sess_w00000", 5.0), _window("sess_w00001", 18.0)]
    return _build_fixture(tmp_path, windows, flac_duration_s=35.0)


# --------------------------------------------------------------------------- #
# gt mode, two windows, fully through ConversationalTTSSystem (the literal
# `python run.py --stages infer` / `--stages measure` dispatch).
# --------------------------------------------------------------------------- #
class TestEndToEndGtViaSystem:
    def test_infer_then_measure_produce_metrics_json(self, two_window_fixture):
        fixture = two_window_fixture
        train_yaml = fixture["tmp_path"] / "train.yaml"
        OmegaConf.save(fixture["training_config"], train_yaml)

        inf_dir = fixture["tmp_path"] / "infer_gt"
        infer_cfg = _infer_config(fixture, "gt", inf_dir)
        infer_cfg.training_config = str(train_yaml)  # absolute -> loaded as-is

        infer_system = ConversationalTTSSystem(inference_config=infer_cfg)
        infer_stats = infer_system.infer()
        assert infer_stats == {"n_selected": 2, "n_skipped": 0}

        test_dir = inf_dir / "valid"
        meta_rows = (test_dir / "meta.scp").read_text("utf-8").splitlines()
        assert len(meta_rows) == 2

        metrics_cfg = _fake_metrics_config(inf_dir)
        measure_system = ConversationalTTSSystem(metrics_config=metrics_cfg)
        results = measure_system.measure()

        metrics_path = inf_dir / "metrics.json"
        assert metrics_path.is_file()
        assert json.loads(metrics_path.read_text("utf-8")) == results
        _assert_all_summary_keys_present(results)


# --------------------------------------------------------------------------- #
# generate mode, through run_inference directly with the tiny DiT + a fake
# Vocos injected (same seam tests/test_inference.py uses). Uses the SAME
# two-window fixture as the gt-mode test above -- a single-window session
# has no non-window prompt candidate for any channel under the reworked
# leakage rule, so both windows must be generated here too.
# --------------------------------------------------------------------------- #
class TestEndToEndGenerateMode:
    def test_infer_then_measure_produce_metrics_json(self, two_window_fixture):
        fixture = two_window_fixture
        inf_dir = fixture["tmp_path"] / "infer_generate"
        infer_cfg = _infer_config(fixture, "generate", inf_dir)
        infer_cfg.sampling.steps = 2  # keep the ODE cheap; plumbing-only smoke

        model = build_tiny(fixture["vocab"]).eval()
        vocoder = FakeVocoder()
        infer_stats = run_inference(
            infer_cfg,
            training_config=fixture["training_config"],
            model=model,
            vocoder=vocoder,
        )
        assert infer_stats == {"n_selected": 2, "n_skipped": 0}

        test_dir = inf_dir / "valid"
        meta = json.loads((test_dir / "meta/sess_w00000.json").read_text("utf-8"))
        assert meta["mode"] == "generate"

        metrics_cfg = _fake_metrics_config(inf_dir)
        results = measure(metrics_cfg)

        metrics_path = inf_dir / "metrics.json"
        assert metrics_path.is_file()
        assert json.loads(metrics_path.read_text("utf-8")) == results
        _assert_all_summary_keys_present(results)
