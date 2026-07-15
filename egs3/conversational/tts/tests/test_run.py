"""Bare ``python run.py`` regression coverage.

Adding ``infer``/``measure`` to the default stages must not break a
no-argument invocation (or ``--stages all``): every default stage's config
has to load from the parser defaults, propagate experiment context, and pass
the required-config guard without raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from egs3.conversational.tts import run
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
)
from espnet3.utils.stages_utils import resolve_stages

RECIPE_DIR = Path(run.__file__).resolve().parent


class TestBareInvocationDefaults:
    def _default_configs(self, monkeypatch):
        """The exact config-resolution path main() runs on parser defaults."""
        monkeypatch.chdir(RECIPE_DIR)  # parser defaults are recipe-relative
        args = run.build_parser(run.DEFAULT_STAGES).parse_args([])
        stages = resolve_stages(args.stages, run.ALL_STAGES)
        training_config = load_and_merge_config(
            args.training_config,
            config_name=run.DEFAULT_TRAINING_CONFIG,
            resolve=False,
        )
        inference_config = load_and_merge_config(
            args.inference_config,
            config_name=run.DEFAULT_INFERENCE_CONFIG,
            resolve=False,
        )
        metrics_config = load_and_merge_config(
            args.metrics_config,
            config_name=run.DEFAULT_METRICS_CONFIG,
            resolve=False,
        )
        return stages, training_config, inference_config, metrics_config

    def test_all_resolves_to_every_stage(self, monkeypatch):
        stages, _, _, _ = self._default_configs(monkeypatch)
        assert stages == run.DEFAULT_STAGES
        assert "infer" in stages
        assert "measure" in stages

    def test_default_configs_load_and_pass_the_guard(self, monkeypatch):
        stages, training_config, inference_config, metrics_config = (
            self._default_configs(monkeypatch)
        )
        assert training_config is not None
        assert inference_config is not None
        assert metrics_config is not None
        # The regression: this raised ValueError while --inference_config
        # defaulted to None with "infer" among the default stages.
        run.check_required_configs(
            stages, training_config, inference_config, metrics_config
        )

    def test_guard_still_raises_when_infer_lacks_its_config(self):
        with pytest.raises(ValueError, match="inference_config"):
            run.check_required_configs(["infer"], object(), None, object())

    def test_guard_still_raises_when_measure_lacks_its_config(self):
        with pytest.raises(ValueError, match="metrics_config"):
            run.check_required_configs(["measure"], object(), object(), None)

    def test_guard_still_raises_when_training_lacks_its_config(self):
        with pytest.raises(ValueError, match="training_config"):
            run.check_required_configs(["train"], None, object(), object())

    def test_metrics_inference_dir_resolves_against_the_real_infer_output(
        self, monkeypatch
    ):
        """End-to-end config-resolution path: after context propagation and
        resolution, metrics_config.inference_dir must land on the SAME
        directory the infer stage actually writes to (not merely "a valid
        string"). The propagation helper reads values with OmegaConf
        ``.get()``, which eagerly resolves interpolations in each config's
        own namespace, so the trap this test guards against is target-side:
        a metrics.yaml ``inference_dir`` formula referencing a key the file
        does not define locally (e.g. ``${mode}`` without a ``mode:`` key)
        raises InterpolationKeyError the moment the propagation compare (or
        final resolution) touches it. conf/metrics.yaml therefore declares
        ``mode`` locally, and this test pins the resulting directory equality
        against the real default configs."""
        stages, training_config, inference_config, metrics_config = (
            self._default_configs(monkeypatch)
        )
        logger = run.configure_logging()
        apply_training_experiment_context(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=None,
            log=logger,
        )
        resolve_loaded_configs(training_config, inference_config, metrics_config)
        assert metrics_config.inference_dir == inference_config.inference_dir


class TestMeasureStageWiring:
    def test_measure_config_default_name(self):
        assert run.DEFAULT_METRICS_CONFIG == "metrics.yaml"

    def test_measure_config_flag_defaults_to_recipe_conf(self):
        parser = run.build_parser(run.DEFAULT_STAGES)
        args = parser.parse_args([])
        assert args.metrics_config == Path("conf/metrics.yaml")
