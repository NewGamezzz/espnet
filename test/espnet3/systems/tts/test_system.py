"""Tests for ESPnet3 TTS system stage hooks.

The TTS system deliberately shares its training helpers with
``espnet3.systems.base.training``.  The only stage it still overrides is
``collect_stats``, and these tests pin down exactly why: the base version pops
``normalize`` / ``normalize_conf`` out of ``config.model``, which for the TTS
task silently restores the ``global_mvn`` default and crashes stats collection.
"""

from omegaconf import OmegaConf

import espnet3.systems.base.training as base_train_mod
import espnet3.systems.tts.system as tts_sysmod
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.tts.system import TTSSystem


class DummyTrainer:
    """Stand-in trainer that records calls instead of building a model."""

    def __init__(self):
        """Initialize the call recorder."""
        self.collect_stats_called = False

    def collect_stats(self):
        """Record that the stats-collection entrypoint was reached."""
        self.collect_stats_called = True


def _make_config(tmp_path):
    """Build a minimal training config carrying an explicit ``normalize: null``.

    ``seed`` and ``parallel`` are intentionally absent so that
    ``_prepare_training_runtime`` skips seeding and parallel setup, keeping the
    test hermetic without patching those out.
    """
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "stats_dir": str(tmp_path / "stats"),
            "model": {
                "normalize": None,
                "normalize_conf": {"stats_file": "should-not-be-needed"},
                "tts": "f5tts",
            },
        }
    )


def test_tts_collect_stats_preserves_normalize_keys(tmp_path, monkeypatch):
    """collect_stats must NOT strip normalize/normalize_conf from the config.

    ``espnet2.tasks.tts`` declares ``normalize_choices`` with
    ``default="global_mvn"``, so dropping ``normalize`` does not disable
    normalization - it re-enables GlobalMVN, which then demands the very
    ``stats_file`` this stage is running to produce.
    """
    cfg = _make_config(tmp_path)
    system = TTSSystem(training_config=cfg)
    trainer = DummyTrainer()

    seen = {}

    def fake_build_trainer(config):
        # Snapshot what the trainer would actually have been handed.
        seen["normalize_present"] = "normalize" in config.model
        seen["normalize_value"] = config.model.get("normalize", "<missing>")
        seen["normalize_conf_present"] = "normalize_conf" in config.model
        return trainer

    monkeypatch.setattr(tts_sysmod, "_build_trainer", fake_build_trainer)

    system.collect_stats()

    assert trainer.collect_stats_called

    # What the trainer saw.
    assert seen["normalize_present"] is True
    assert seen["normalize_value"] is None
    assert seen["normalize_conf_present"] is True

    # And the config was not mutated as a side effect either.
    assert "normalize" in cfg.model
    assert cfg.model.normalize is None
    assert "normalize_conf" in cfg.model


def test_tts_collect_stats_creates_directories(tmp_path, monkeypatch):
    """The inherited ``_ensure_directories`` still runs via the TTS override."""
    cfg = _make_config(tmp_path)
    system = TTSSystem(training_config=cfg)

    monkeypatch.setattr(tts_sysmod, "_build_trainer", lambda _cfg: DummyTrainer())

    system.collect_stats()

    assert (tmp_path / "exp").is_dir()
    assert (tmp_path / "stats").is_dir()


def test_base_collect_stats_pops_normalize_keys(tmp_path, monkeypatch):
    """Contrast case: the base stage DOES pop the keys.

    This is the behaviour ``TTSSystem.collect_stats`` exists to avoid.  If this
    test ever starts failing because the base stopped popping, the TTS override
    can be reconsidered - but not before.
    """
    cfg = _make_config(tmp_path)
    trainer = DummyTrainer()

    monkeypatch.setattr(base_train_mod, "_build_trainer", lambda _cfg: trainer)
    monkeypatch.setattr(
        base_train_mod.torch, "set_float32_matmul_precision", lambda _p: None
    )

    base_train_mod.collect_stats(cfg)

    assert trainer.collect_stats_called
    assert "normalize" not in cfg.model
    assert "normalize_conf" not in cfg.model


def test_tts_system_inherits_base_train(tmp_path):
    """``train`` must come from ``BaseSystem``; no duplicated override."""
    assert "train" not in TTSSystem.__dict__
    assert TTSSystem.train is BaseSystem.train


def test_tts_system_reuses_base_training_helpers():
    """The trainer/dir helpers are the shared base objects, not local copies."""
    assert tts_sysmod._build_trainer is base_train_mod._build_trainer
    assert tts_sysmod._ensure_directories is base_train_mod._ensure_directories
    assert not hasattr(TTSSystem, "_build_trainer")
    assert not hasattr(TTSSystem, "_ensure_directories")


def test_tts_system_still_overrides_collect_stats():
    """collect_stats is the one stage TTS must keep for itself."""
    assert "collect_stats" in TTSSystem.__dict__
    assert TTSSystem.collect_stats is not BaseSystem.collect_stats
