"""Bare ``python run.py`` regression coverage.

Adding ``infer`` to the default stages must not break a no-argument invocation
(or ``--stages all``): every default stage's config has to load from the
parser defaults and pass the required-config guard without raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from egs3.conversational.tts import run
from espnet3.utils.config_utils import load_and_merge_config
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
        return stages, training_config, inference_config

    def test_all_resolves_to_every_stage(self, monkeypatch):
        stages, _, _ = self._default_configs(monkeypatch)
        assert stages == run.DEFAULT_STAGES
        assert "infer" in stages

    def test_default_configs_load_and_pass_the_guard(self, monkeypatch):
        stages, training_config, inference_config = self._default_configs(monkeypatch)
        assert training_config is not None
        assert inference_config is not None
        # The regression: this raised ValueError while --inference_config
        # defaulted to None with "infer" among the default stages.
        run.check_required_configs(stages, training_config, inference_config)

    def test_guard_still_raises_when_infer_lacks_its_config(self):
        with pytest.raises(ValueError, match="inference_config"):
            run.check_required_configs(["infer"], object(), None)

    def test_guard_still_raises_when_training_lacks_its_config(self):
        with pytest.raises(ValueError, match="training_config"):
            run.check_required_configs(["train"], None, object())
