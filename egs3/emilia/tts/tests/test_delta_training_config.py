"""Training configs for NCSA Delta (4 x A100-40GB).

``training_f5_tts_base_delta.yaml`` is the Base run moved from PSC's V100s:
the same experiment and the same 307,200 frames per update, with a larger
micro-batch in bf16. ``training_f5_tts_base_delta_trial.yaml`` is a short copy
of it for the interactive partition.

Each config is a full standalone file (run.py merges a config over the
near-empty template, never over another recipe config), so these tests are
what keep the copies faithful.
"""

from omegaconf import OmegaConf

from egs3.emilia.tts.tests.test_training_config import (
    CONF,
    DELTA_CONF,
    DELTA_TRIAL_CONF,
    _flatten,
)


def _resolved(path):
    return _flatten(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def test_delta_config_differs_from_base_only_in_the_hardware_knobs():
    """Same experiment as PSC: only what the A100s change may differ.

    In particular ``exp_tag`` is untouched, so the PSC checkpoint resumes in
    the directory and logger run it was written for.
    """
    base = _resolved(CONF)
    delta = _resolved(DELTA_CONF)

    assert set(delta) - set(base) == {"trainer.precision"}
    assert set(base) - set(delta) == set()
    differing = {k for k in base if base[k] != delta[k]}
    assert differing == {
        "dataloader.train.iter_factory.batches.batch_bins",
        "dataloader.train.iter_factory.num_workers",
        "trainer.accumulate_grad_batches",
        "trainer.val_check_interval",
    }


def test_delta_config_keeps_the_307200_frame_update():
    cfg = OmegaConf.load(DELTA_CONF)
    bins = cfg.dataloader.train.iter_factory.batches.batch_bins
    accum = cfg.trainer.accumulate_grad_batches
    assert bins * accum * cfg.num_device / cfg.n_mel_channels == 307200
    assert bins > OmegaConf.load(CONF).dataloader.train.iter_factory.batches.batch_bins


def test_delta_config_trains_in_bf16_mixed():
    assert OmegaConf.load(DELTA_CONF).trainer.precision == "bf16-mixed"


def test_delta_config_validates_on_the_batches_psc_validates_on():
    """A larger train batch must not change what the validation loss covers.

    The valid sampler packs its batches by ``batch_bins`` and validation stops
    after ``limit_val_batches``, so a valid budget that followed the train one
    would score different utterances than the PSC run did. It is written as a
    number here, not as the interpolation the base config uses.

    This pins the budget only. The utterances are the same only when the
    valid manifest is: the builder draws the split from a seeded shuffle over
    all kept rows, so another corpus copy yields another split.
    """
    base = OmegaConf.load(CONF)
    delta = OmegaConf.load(DELTA_CONF)
    raw = OmegaConf.to_container(delta, resolve=False)
    valid_bins = raw["dataloader"]["valid"]["iter_factory"]["batches"]["batch_bins"]
    assert valid_bins == base.dataloader.train.iter_factory.batches.batch_bins
    assert delta.trainer.limit_val_batches == base.trainer.limit_val_batches


def test_delta_config_validates_at_the_same_optimizer_steps_as_psc():
    """Lightning counts ``val_check_interval`` in micro-batches, not updates.

    Fewer micro-batches per update therefore need a proportionally smaller
    interval for validation to land on the same optimizer steps.
    """
    base = OmegaConf.load(CONF).trainer
    delta = OmegaConf.load(DELTA_CONF).trainer
    assert (
        delta.val_check_interval / delta.accumulate_grad_batches
        == base.val_check_interval / base.accumulate_grad_batches
    )


def test_delta_trial_differs_from_the_delta_config_only_in_length_and_logging():
    """The trial measures the production step, so every per-step knob stays."""
    delta = _resolved(DELTA_CONF)
    trial = _resolved(DELTA_TRIAL_CONF)

    def rest(flat):
        return {k: v for k, v in flat.items() if not k.startswith("trainer.logger")}

    delta_rest, trial_rest = rest(delta), rest(trial)
    assert set(delta_rest) == set(trial_rest)
    differing = {k for k in delta_rest if delta_rest[k] != trial_rest[k]}
    assert differing == {
        "exp_tag",
        # both follow from exp_tag
        "exp_dir",
        "inference_dir",
        "trainer.max_steps",
        "trainer.val_check_interval",
        "trainer.limit_val_batches",
        "trainer.save_every_n_train_steps",
    }
    assert trial["stats_dir"] == delta["stats_dir"]


def test_delta_trial_never_writes_into_the_production_experiment():
    """A leftover trial checkpoint must not be what a later resume picks up."""
    delta = OmegaConf.load(DELTA_CONF)
    trial = OmegaConf.load(DELTA_TRIAL_CONF)
    assert trial.exp_tag != delta.exp_tag
    assert trial.exp_dir != delta.exp_dir
    # offline: a trial leaves no run behind in the production logger project
    assert trial.trainer.logger._target_ == "lightning.pytorch.loggers.CSVLogger"


def test_delta_trial_checkpoints_and_validates_before_it_ends():
    """One short run has to exercise the save and the validation path too."""
    trainer = OmegaConf.load(DELTA_TRIAL_CONF).trainer
    assert 0 < trainer.max_steps <= 1000
    assert trainer.save_every_n_train_steps * 2 <= trainer.max_steps
    first_validation = trainer.val_check_interval / trainer.accumulate_grad_batches
    assert first_validation < trainer.max_steps
    assert 0 < trainer.limit_val_batches <= 20
