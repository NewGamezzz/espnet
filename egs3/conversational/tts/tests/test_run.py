"""Bare ``python run.py`` regression coverage.

Adding ``infer``/``measure`` to the default stages must not break a
no-argument invocation (or ``--stages all``): every default stage's config
has to load from the parser defaults, propagate experiment context, and pass
the required-config guard without raising.
"""

from __future__ import annotations

import re
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


class TestAnchorModePropagation:
    """Context propagation when inference_config.mode != metrics_config.mode.

    ``_copy_config_context`` OVERWRITES a differing target value with the
    source's (resolved) value, logging only a warning -- it does not skip on
    conflict. Since ``run.py`` always loads an inference config (defaulting
    to the shipped generate-mode file), a ``--stages measure`` run given only
    a mode-edited metrics config would get its ``inference_dir`` silently
    replaced by ``infer_generate``. These tests pin BOTH sides of that
    behavior against the real shipped configs: the corrected anchor
    invocation (matching mode-edited ``--inference_config`` passed too)
    scores the anchor dir, and the flag-omitting invocation demonstrably
    does not -- which is exactly why README.md's anchor loop passes
    ``--inference_config`` to the measure line.
    """

    def _load_and_propagate(self, monkeypatch, tmp_path, inference_mode):
        """main()'s load/apply/resolve sequence with mode-edited config
        copies (the sed loop from README.md's anchor recipe): metrics is
        always the gt copy; ``inference_mode`` picks the inference config."""
        monkeypatch.chdir(RECIPE_DIR)

        def _mode_copy(src_name: str, out_name: str, mode: str) -> Path:
            # Line-anchored, exactly like the README's
            # `sed "s/^mode: generate/mode: $m/"` (a bare replace would also
            # rewrite the phrase inside metrics.yaml's header comment).
            src = RECIPE_DIR / "conf" / src_name
            out = tmp_path / out_name
            out.write_text(
                re.sub(
                    r"^mode: generate",
                    f"mode: {mode}",
                    src.read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                ),
                encoding="utf-8",
            )
            return out

        metrics_path = _mode_copy("metrics.yaml", "metrics_gt.yaml", "gt")
        if inference_mode == "generate":
            inference_path = RECIPE_DIR / "conf" / "inference_conversational.yaml"
        else:
            inference_path = _mode_copy(
                "inference_conversational.yaml",
                f"inference_{inference_mode}.yaml",
                inference_mode,
            )

        training_config = load_and_merge_config(
            Path("conf/training_poc.yaml"),
            config_name=run.DEFAULT_TRAINING_CONFIG,
            resolve=False,
        )
        # tmp_path copies sit outside egs3/, so the TEMPLATE package cannot
        # be inferred from the path and is passed explicitly.
        inference_config = load_and_merge_config(
            inference_path,
            config_name=run.DEFAULT_INFERENCE_CONFIG,
            resolve=False,
            default_package="egs3.TEMPLATE.tts",
        )
        metrics_config = load_and_merge_config(
            metrics_path,
            config_name=run.DEFAULT_METRICS_CONFIG,
            resolve=False,
            default_package="egs3.TEMPLATE.tts",
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
        return inference_config, metrics_config

    def test_matching_mode_configs_keep_the_anchor_inference_dir(
        self, monkeypatch, tmp_path
    ):
        # The corrected README loop: measure passes BOTH mode-edited configs.
        inference_config, metrics_config = self._load_and_propagate(
            monkeypatch, tmp_path, inference_mode="gt"
        )
        assert str(metrics_config.inference_dir).endswith("infer_gt")
        assert metrics_config.inference_dir == inference_config.inference_dir

    def test_omitting_the_inference_config_flag_scores_the_wrong_condition(
        self, monkeypatch, tmp_path
    ):
        # The documented failure mode: metrics says gt, but the DEFAULT
        # (generate-mode) inference config wins the overwrite-on-differ
        # propagation, so measure would silently score infer_generate.
        inference_config, metrics_config = self._load_and_propagate(
            monkeypatch, tmp_path, inference_mode="generate"
        )
        assert str(metrics_config.inference_dir).endswith("infer_generate")
        assert metrics_config.inference_dir == inference_config.inference_dir


def test_ami_configs_load_and_agree():
    """conf/inference_ami.yaml + conf/metrics_ami.yaml: the SSSD-path AMI
    configs (design note 'Beyond Two Speakers Evaluation on AMI', section 5)."""
    from omegaconf import OmegaConf

    recipe = Path(__file__).resolve().parents[1]
    inf = OmegaConf.load(recipe / "conf" / "inference_ami.yaml")
    met = OmegaConf.load(recipe / "conf" / "metrics_ami.yaml")
    assert inf.mode == met.mode == "generate"
    assert inf.dataset.split == "ami_test"
    assert inf.dataset.manifest_path.endswith("manifest/ami_test.jsonl")
    assert inf.prompt.solo_guard_sec == 0.3
    assert inf.prompt.exclude_spans.endswith("exclude_spans.json")
    assert inf.anchor.mask_to_turns.enabled is True
    assert inf.anchor.mask_to_turns.guard_sec == 0.15
    assert (inf.sampling.steps, inf.sampling.cfg_strength, inf.sampling.sway_sampling_coef) == (
        64, 3.0, -1.0,
    )
    assert inf.duration.source == "predicted" and inf.duration.rate_prior_chars == 100.0
    assert inf.duration.scale == 1.048
    assert inf.selection.manifest is None and inf.selection.num_active_speakers == 2
    assert inf.selection.per_session_cap == 12
    assert inf.text_format == "order"
    assert met.dataset.test[0].name == inf.test_name == "ami_k2"


def test_ami_longform_configs_load_and_agree():
    """conf/inference_ami_longform_chunked.yaml + conf/metrics_ami_longform.yaml:
    the chunked external-path AMI long-form arm (one meeting per dialogue)."""
    from omegaconf import OmegaConf

    recipe = Path(__file__).resolve().parents[1]
    inf = OmegaConf.load(recipe / "conf" / "inference_ami_longform_chunked.yaml")
    met = OmegaConf.load(recipe / "conf" / "metrics_ami_longform.yaml")
    assert inf.mode == met.mode == "generate_external_chunked"
    assert inf.testset.manifest.endswith("ami-longform-v1/manifest.jsonl")
    assert inf.prompt_fill == "room_tone"
    assert inf.chunk.unchunked_max_sec is None and inf.chunk.turns is None
    assert inf.chunk.target_sec == 25.0 and inf.chunk.cover_all_speakers is True
    assert inf.chunk.cond_silence_gate is True and inf.chunk.cond_loudness_norm is True
    assert inf.chunk.cross_fade_sec == 0.1
    assert (inf.sampling.cfg_strength, inf.sampling.cfg_sparse_strength, inf.sampling.cfg_sparse_max_chars) == (3.0, 2.0, 40)
    assert inf.duration.source == "predicted" and inf.duration.rate_prior_chars == 100.0
    assert inf.batching.max_batch_audio_sec == 120.0
    raw_inf = OmegaConf.to_container(inf, resolve=False)
    raw_met = OmegaConf.to_container(met, resolve=False)
    assert raw_met["inference_dir"] == raw_inf["inference_dir"] == "${exp_dir}/ami_longform_cover"
    assert met.dataset.test[0].name == inf.test_name == "valid"
