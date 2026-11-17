"""Training configs for NCSA Delta (4 x A100-40GB).

``training_f5_tts_base_delta.yaml`` trains the Base model from scratch on
Delta: the base config's model, optimizer, schedule and 307,200 frames per
update, with a larger micro-batch in bf16, as its own experiment, and with the
pass over the data cut into short epochs so that a chain of jobs resumes
exactly. ``training_f5_tts_base_delta_trial.yaml`` is a short copy of it for
the interactive partition.

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

# keys that only repeat exp_tag
FOLLOWS_EXP_TAG = {
    "exp_dir",
    "inference_dir",
    "trainer.logger.name",
    "trainer.logger.id",
    "trainer.logger.save_dir",
}
TRAIN_FACTORY = "dataloader.train.iter_factory"


def _resolved(path):
    return _flatten(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def test_delta_config_differs_from_base_only_in_hardware_tag_and_epoch_slicing():
    """Model, optimizer, schedule and data settings are the base config's."""
    base = _resolved(CONF)
    delta = _resolved(DELTA_CONF)

    assert set(delta) - set(base) == {
        "trainer.precision",
        "trainer.check_val_every_n_epoch",
        f"{TRAIN_FACTORY}.num_iters_per_epoch",
        f"{TRAIN_FACTORY}.seed",
    }
    assert set(base) - set(delta) == {
        "trainer.save_every_n_train_steps",
        "trainer.val_check_interval",
    }
    differing = {k for k in base if k in delta and base[k] != delta[k]}
    assert differing == {
        "exp_tag",
        *FOLLOWS_EXP_TAG,
        f"{TRAIN_FACTORY}.batches.batch_bins",
        f"{TRAIN_FACTORY}.num_workers",
        "trainer.accumulate_grad_batches",
    }


def test_delta_config_is_its_own_experiment():
    """A run from scratch must not log into, or resume, the base config's run.

    The logger resumes the run whose id is ``exp_tag``, and the chain resumes
    whatever checkpoint is in ``exp_dir``.
    """
    base = OmegaConf.load(CONF)
    delta = OmegaConf.load(DELTA_CONF)
    assert delta.exp_tag != base.exp_tag
    assert delta.exp_dir != base.exp_dir
    assert delta.trainer.logger.id == delta.exp_tag


def test_delta_config_keeps_the_307200_frame_update():
    cfg = OmegaConf.load(DELTA_CONF)
    bins = cfg.dataloader.train.iter_factory.batches.batch_bins
    accum = cfg.trainer.accumulate_grad_batches
    assert bins * accum * cfg.num_device / cfg.n_mel_channels == 307200
    assert bins > OmegaConf.load(CONF).dataloader.train.iter_factory.batches.batch_bins


def test_delta_config_trains_in_bf16_mixed():
    assert OmegaConf.load(DELTA_CONF).trainer.precision == "bf16-mixed"


def test_delta_config_keeps_the_base_validation_budget():
    """A larger train batch must not change how validation batches are packed.

    The valid sampler packs its batches by ``batch_bins`` and validation stops
    after ``limit_val_batches``, so a valid budget that followed the train one
    would score other utterances. It is written as a number here, not as the
    interpolation the base config uses.

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


def test_delta_config_checkpoints_at_the_end_of_short_epochs_only():
    """A chain of jobs resumes exactly only from an epoch-end checkpoint.

    The loader cannot resume in the middle of an epoch: after a step-based
    checkpoint it starts again at the epoch's first batch while the step
    counter goes on, so every job of a chain trains the same opening batches
    and the rest of the pass is never reached. The pass is therefore cut into
    slices (``num_iters_per_epoch``), each an epoch that ends with a
    validation and a checkpoint, and nothing is saved in between.
    """
    cfg = OmegaConf.load(DELTA_CONF)
    trainer = cfg.trainer
    factory = cfg.dataloader.train.iter_factory
    assert "save_every_n_train_steps" not in trainer
    assert "val_check_interval" not in trainer
    # without a validation at every epoch end Lightning repeats a batch on resume
    assert trainer.check_val_every_n_epoch == 1
    assert trainer.max_epochs == -1
    assert factory.num_iters_per_epoch > 0
    # an epoch that ends inside an accumulation window takes a partial step
    assert factory.num_iters_per_epoch % trainer.accumulate_grad_batches == 0
    # epoch 0 draws RandomState(seed - 1) for the pass before the first
    assert factory.seed >= 1


def test_delta_trial_differs_from_the_delta_config_only_in_tag_and_length():
    """The trial exercises the production step, loader, logger and resume."""
    delta = _resolved(DELTA_CONF)
    trial = _resolved(DELTA_TRIAL_CONF)

    assert set(delta) == set(trial)
    differing = {k for k in delta if delta[k] != trial[k]}
    assert differing == {
        "exp_tag",
        *FOLLOWS_EXP_TAG,
        "trainer.max_steps",
        "trainer.limit_val_batches",
        f"{TRAIN_FACTORY}.num_iters_per_epoch",
    }
    assert trial["stats_dir"] == delta["stats_dir"]


def test_delta_trial_never_writes_into_the_production_experiment():
    """A leftover trial checkpoint must not be what a later resume picks up."""
    delta = OmegaConf.load(DELTA_CONF)
    trial = OmegaConf.load(DELTA_TRIAL_CONF)
    assert trial.exp_tag != delta.exp_tag
    assert trial.exp_dir != delta.exp_dir
    assert trial.trainer.logger.id == trial.exp_tag


def test_delta_trial_ends_several_epochs_within_a_short_job():
    """Two short jobs have to show epoch-end saves and a resume between them."""
    cfg = OmegaConf.load(DELTA_TRIAL_CONF)
    trainer = cfg.trainer
    slice_len = cfg.dataloader.train.iter_factory.num_iters_per_epoch
    assert slice_len % trainer.accumulate_grad_batches == 0
    updates_per_epoch = slice_len // trainer.accumulate_grad_batches
    assert updates_per_epoch <= 200
    assert trainer.max_steps >= 10 * updates_per_epoch
    assert 0 < trainer.limit_val_batches <= 20
