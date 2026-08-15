"""The Base training config matches the paper and the spec's decisions."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egs3.emilia.tts.tests._dotted_paths import iter_targets, resolve_dotted_path

CONF = Path(__file__).resolve().parents[1] / "conf" / "training_f5_tts_base.yaml"
SMOKE_CONF = (
    Path(__file__).resolve().parents[1] / "conf" / "training_f5_tts_base_smoke.yaml"
)
SMOKE_2GPU_CONF = (
    Path(__file__).resolve().parents[1]
    / "conf"
    / "training_f5_tts_base_smoke_2gpu.yaml"
)


@pytest.fixture
def cfg():
    return OmegaConf.load(CONF)


def _flatten(container, prefix=""):
    """Flatten a nested dict/list into {dotted.path: leaf_value}."""
    flat = {}
    if isinstance(container, dict):
        items = container.items()
    elif isinstance(container, list):
        items = enumerate(container)
    else:
        return {prefix: container}
    for key, value in items:
        path = f"{prefix}.{key}" if prefix else str(key)
        flat.update(_flatten(value, path))
    return flat


def test_base_architecture_matches_paper(cfg):
    tts = cfg.model.tts_conf
    assert (tts.dim, tts.depth, tts.heads) == (1024, 22, 16)
    assert tts.dim_head == 64
    assert tts.ff_mult == 2
    assert tts.text_dim == 512
    assert tts.conv_layers == 4


def test_global_batch_is_307200_frames(cfg):
    """batch_bins * accum * n_gpu / n_mels == 307200 (spec 6.1)."""
    bins = cfg.dataloader.train.iter_factory.batches.batch_bins
    accum = cfg.trainer.accumulate_grad_batches
    n_gpu = cfg.num_device
    n_mels = cfg.n_mel_channels
    assert bins * accum * n_gpu / n_mels == 307200


def test_optimizer_and_schedule_match_paper(cfg):
    assert cfg.optimizer.lr == 7.5e-5
    assert cfg.scheduler.warmup_steps == 20000
    assert cfg.scheduler.total_steps == 1200000
    assert cfg.trainer.max_steps == 1200000
    assert cfg.trainer.gradient_clip_val == 1.0


def test_precision_is_fp32(cfg):
    """D5: V100 has no bf16 and every verified result is fp32."""
    assert "precision" not in cfg.trainer


def test_validation_is_step_based(cfg):
    """Spec 6.2: 12 epochs total, so per-epoch validation is useless.

    Pin the actual values, not just `> 0`: 5000/200 are a named spec
    decision (spec 6.2), not incidental numbers a `> 0` check would let
    drift silently.
    """
    assert "check_val_every_n_epoch" not in cfg.trainer
    assert cfg.trainer.val_check_interval == 5000
    assert cfg.trainer.limit_val_batches == 200


def test_uses_f5_pinyin_preprocessor_not_common(cfg):
    assert cfg.dataset.preprocessor._target_ == (
        "espnet2.text.f5_preprocessor.F5PinyinPreprocessor"
    )


def test_normalize_is_null_and_no_collect_stats(cfg):
    assert cfg.model.normalize is None
    assert "collect_stats" not in cfg
    assert "remove_long_short" not in cfg
    assert "create_token_list" not in cfg


def test_fit_has_no_unconditional_ckpt_path(cfg):
    """IMPORTANT 1: `fit.ckpt_path: ${exp_dir}/last.ckpt` unconditionally
    FileNotFoundErrors on a fresh run -- verified directly against the
    installed lightning==2.6.5: CheckpointConnector._parse_ckpt_path passes
    a literal, non-"best"/"last"/"hpc" ckpt_path straight through to
    TorchCheckpointIO.load_checkpoint, which raises FileNotFoundError for a
    nonexistent path (lightning_fabric/plugins/io/torch_io.py:88-89).

    Resume is instead supplied conditionally by local/submit_train.sbatch
    via `run.py --ckpt_path`, only when $LAST_CKPT already exists -- see
    tests/test_sbatch_scripts.py and tests/test_run_ckpt_path_override.py.
    """
    assert cfg.fit == {}


def test_recipe_dir_is_set(cfg):
    """create_shape reads training_config.recipe_dir by plain attribute
    access with no fallback (src/system.py), so this pins the contract."""
    assert cfg.recipe_dir is not None

    # Check raw (unresolved) interpolation strings, not resolved values:
    # data_dir/exp_dir must reference ${recipe_dir} directly. stats_dir is
    # intentionally NOT scoped through exp_dir (see
    # test_stats_dir_is_shared_across_exp_tags): it chains straight back to
    # ${recipe_dir} instead, since feats_shape is a property of the corpus,
    # not of one experiment.
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert "${recipe_dir}" in raw["data_dir"]
    assert "${recipe_dir}" in raw["exp_dir"]
    assert "${recipe_dir}" in raw["stats_dir"]
    assert "${exp_dir}" not in raw["stats_dir"]


def test_stats_dir_is_shared_across_exp_tags(cfg):
    """CRITICAL 3: stats_dir (and everything derived from it) must be
    independent of exp_tag, or create_shape run against one exp_tag writes
    feats_shape somewhere a different exp_tag's dataloader never looks --
    exactly what happened when both training_f5_tts_base.yaml and
    training_f5_tts_base_smoke.yaml derived stats_dir from ${exp_dir}."""
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert "${exp_tag}" not in raw["stats_dir"]
    assert "${exp_dir}" not in raw["stats_dir"]


@pytest.mark.parametrize("conf_path", [CONF, SMOKE_CONF])
def test_stats_dir_survives_the_real_template_merge(conf_path):
    """CRITICAL 3, through the actual production pipeline, not a plain
    OmegaConf.load of the recipe file in isolation.

    egs3/TEMPLATE/tts/conf/training.yaml (the near-empty default every
    recipe config merges over -- see run.py's DEFAULT_TRAINING_CONFIG,
    fixed alongside this in the same review wave) itself defines
    `stats_dir: ${exp_dir}/stats`. Before that DEFAULT_TRAINING_CONFIG fix,
    load_and_merge_config always raised FileNotFoundError, so this merge
    never actually happened in production; fixing it newly activates a
    merge this test needs to check, not assume. OmegaConf.merge(default,
    user) means the recipe's own stats_dir should win over the template's,
    but "should" is exactly what CRITICAL 3 was about -- so resolve
    through the real load_and_merge_config, the same call
    local/submit_train.sbatch and run.py both make, and confirm the result
    is still exp_tag-independent."""
    from espnet3.utils.config_utils import load_and_merge_config

    cfg = load_and_merge_config(conf_path, config_name="training.yaml")
    assert "exp_tag" not in cfg.stats_dir
    assert cfg.stats_dir.endswith("/exp/stats")


def test_create_shape_block_has_manifest_paths(cfg):
    """src/system.py:100 does cfg["manifest_paths"][split] and would
    KeyError without both splits present."""
    manifest_paths = cfg.create_shape.manifest_paths
    assert "train" in manifest_paths
    assert "valid" in manifest_paths


def test_train_uses_numel_array_with_upstream_max_samples_cap(cfg):
    """Task 12: NumElementsArraySampler with upstream's max_samples=64 cap,
    fixing the 300+-sample batches an uncapped numel sampler would produce
    at the short end of Emilia's length-sorted order (spec 6.1)."""
    batches = cfg.dataloader.train.iter_factory.batches
    assert batches.type == "numel_array"
    assert batches.max_samples == 64


def test_valid_also_uses_numel_array_with_max_samples_cap(cfg):
    """Valid inherits type via interpolation from train; max_samples is set
    explicitly too. With sort_batch's ascending default, validation's first
    limit_val_batches batches are exactly the short-utterance region where
    an uncapped batch would be largest, so the same OOM risk as train
    applies to validation, just earlier."""
    batches = cfg.dataloader.valid.iter_factory.batches
    assert batches.type == "numel_array"
    assert batches.max_samples == 64


def test_smoke_config_only_differs_from_base_in_five_places():
    """The smoke config exists to measure PRODUCTION throughput, so it must
    be the Base config with exactly max_steps, val_check_interval,
    limit_val_batches, exp_tag and trainer.logger changed -- in particular
    batch_bins, accumulate_grad_batches and num_device must be untouched.

    load_and_merge_config merges each config over the near-empty
    egs3.TEMPLATE.tts default, never over another recipe config (see
    run.py's comment), so the smoke config is a full standalone copy rather
    than a thin override; this test is what actually enforces "everything
    else stays at production values" instead of relying on the copy being
    faithful by inspection.

    Compares RESOLVED values, not raw interpolation strings (CRITICAL 3
    regression guard). The original version of this test used
    `resolve=False`, so `stats_dir: ${exp_dir}/stats` was a byte-identical
    *string* in both configs and the test passed precisely where the two
    configs actually diverged: with exp_tag interpolated in, base resolved
    to exp/train_f5_tts_base_emilia/stats while smoke resolved to
    exp/smoke_base_emilia/stats -- two different, unrelated directories.
    Resolving first is what makes this test able to catch that class of
    bug; see test_stats_dir_is_shared_across_exp_tags for the direct fix
    pin and the falsification note below.
    """
    base = _flatten(OmegaConf.to_container(OmegaConf.load(CONF), resolve=True))
    smoke = _flatten(
        OmegaConf.to_container(OmegaConf.load(SMOKE_CONF), resolve=True)
    )

    # trainer.logger is a wholesale block replacement (WandbLogger ->
    # CSVLogger), so its key SET is expected to differ, not just its
    # values; that structural swap is checked separately below and in
    # test_smoke_config_uses_offline_csv_logger. Everything else must have
    # an identical key set.
    base_rest = {k: v for k, v in base.items() if not k.startswith("trainer.logger")}
    smoke_rest = {k: v for k, v in smoke.items() if not k.startswith("trainer.logger")}
    assert set(base_rest) == set(smoke_rest), "smoke config adds/drops keys vs. base"

    differing = {k for k in base_rest if base_rest[k] != smoke_rest[k]}
    expected = {
        "exp_tag",
        # exp_dir and inference_dir both interpolate ${exp_tag}, so they are
        # legitimate, expected consequences of the single exp_tag change --
        # NOT independent differences. Widening this set beyond that (e.g.
        # to also exclude stats_dir) would re-hide the exact bug this test
        # exists to catch; stats_dir is deliberately NOT in this set (see
        # the explicit assertion below).
        "exp_dir",
        "inference_dir",
        "trainer.max_steps",
        "trainer.val_check_interval",
        "trainer.limit_val_batches",
    }
    assert differing == expected

    # The logger block itself must differ (WandbLogger -> CSVLogger); its
    # exact shape is pinned by test_smoke_config_uses_offline_csv_logger.
    assert base["trainer.logger._target_"] != smoke["trainer.logger._target_"]

    # Direct, explicit pin (CRITICAL 3): stats_dir and everything derived
    # from it must be IDENTICAL between base and smoke once resolved, since
    # only training_f5_tts_base.yaml's create_shape stage actually runs
    # (the smoke's own README-documented sequence never calls create_shape)
    # and the smoke's dataloader must read what that run wrote.
    assert base["stats_dir"] == smoke["stats_dir"]
    assert base["create_shape.save_path"] == smoke["create_shape.save_path"]
    assert (
        base["dataloader.train.iter_factory.batches.shape_files.0"]
        == smoke["dataloader.train.iter_factory.batches.shape_files.0"]
    )
    assert (
        base["dataloader.valid.iter_factory.batches.shape_files.0"]
        == smoke["dataloader.valid.iter_factory.batches.shape_files.0"]
    )


def test_smoke_config_uses_offline_csv_logger():
    """logger: false can never work in espnet3: default_callbacks.py installs
    a LearningRateMonitor() unconditionally, and Lightning refuses that
    callback without a logger. CSVLogger is offline (local files only, no
    wandb run left behind by a one-off smoke test)."""
    cfg = OmegaConf.load(SMOKE_CONF)
    assert cfg.trainer.logger._target_ == "lightning.pytorch.loggers.CSVLogger"


def test_smoke_config_keeps_production_batch_arithmetic():
    """Same 307200-frame global batch as Base -- this is the entire point of
    measuring throughput on the smoke run."""
    cfg = OmegaConf.load(SMOKE_CONF)
    bins = cfg.dataloader.train.iter_factory.batches.batch_bins
    accum = cfg.trainer.accumulate_grad_batches
    n_gpu = cfg.num_device
    n_mels = cfg.n_mel_channels
    assert bins * accum * n_gpu / n_mels == 307200


def test_smoke_config_has_short_step_budget():
    cfg = OmegaConf.load(SMOKE_CONF)
    assert cfg.trainer.max_steps == 200
    assert cfg.trainer.val_check_interval == 100
    assert cfg.trainer.limit_val_batches == 5
    assert cfg.exp_tag == "smoke_base_emilia"


# --- IMPORTANT 5, training configs (advisor follow-up on the final
# whole-branch review): test_seedtts_inference_config.py and
# test_metrics_config.py close this gap for the infer/measure configs, but
# left the training configs' own dotted paths -- task, dataset._target_,
# optimizer._target_, scheduler._target_, dataloader.collate_fn._target_,
# both iter_factory._target_s, trainer.callbacks[0]._target_,
# trainer.logger._target_ -- unresolved by any test. --dry_run never
# instantiates any of these (run_stages hits `if dry_run: continue` before
# calling the stage), so a wrong path here would first fail at trainer
# construction on a real PSC job, exactly the class of defect C1/C2 were.


@pytest.mark.parametrize("conf_path", [CONF, SMOKE_CONF])
def test_task_and_every_target_resolve(conf_path):
    cfg = OmegaConf.load(conf_path)
    resolved = OmegaConf.to_container(cfg, resolve=True)

    # `task` isn't a `_target_`, but is resolved the same way in production
    # (espnet3/utils/task_utils.py's get_task_class -> hydra.utils.get_class,
    # semantically the same import+getattr resolve_dotted_path performs).
    task_obj = resolve_dotted_path(resolved["task"])
    assert task_obj is not None

    targets = list(iter_targets(resolved))
    # Sanity: confirm the walk actually found the targets this test exists
    # to check, so a future edit that silently drops one doesn't make this
    # test vacuously pass on fewer paths checked.
    expected_min = {
        "espnet3.components.data.data_organizer.DataOrganizer",
        "espnet2.text.f5_preprocessor.F5PinyinPreprocessor",
        "torch.optim.AdamW",
        "espnet2.schedulers.linear_warmup_decay.linear_warmup_decay",
        "espnet2.train.collate_fn.CommonCollateFn",
        "espnet2.iterators.sequence_iter_factory.SequenceIterFactory",
        "espnet3.components.callbacks.ema.EMACallback",
    }
    assert expected_min <= set(targets), sorted(expected_min - set(targets))
    for target in targets:
        resolve_dotted_path(target)


def test_base_and_smoke_logger_targets_both_resolve():
    """trainer.logger._target_ is the one target that structurally differs
    between base (WandbLogger) and smoke (CSVLogger); pin both explicitly
    since test_task_and_every_target_resolve's `expected_min` check doesn't
    distinguish which config produced which logger target."""
    base_logger = OmegaConf.load(CONF).trainer.logger._target_
    smoke_logger = OmegaConf.load(SMOKE_CONF).trainer.logger._target_
    assert base_logger == "lightning.pytorch.loggers.WandbLogger"
    assert smoke_logger == "lightning.pytorch.loggers.CSVLogger"
    resolve_dotted_path(base_logger)
    resolve_dotted_path(smoke_logger)


def test_smoke_2gpu_config_changes_only_device_count_and_paths(monkeypatch):
    """The 2-GPU smoke must keep every per-rank knob at production values.

    Its whole purpose is to answer "does Base fit on a V100-32" and "how many
    updates per hour" on 2 GPUs instead of 8. Both answers only transfer if
    the per-rank work is identical to production: same batch_bins, same
    min_batch_size on both loaders, and above all the same
    accumulate_grad_batches, since time per optimizer update is
    accum micro-steps plus the allreduce. Bumping accum to 16 to restore the
    307,200-frame global batch would quadruple micro-steps per update and
    silently destroy the throughput comparison, so this test forbids it.

    Consequence, asserted explicitly below: the 2-GPU config's global batch
    is 76,800 frames, NOT 307,200. That makes it an instrument and not a
    training config, which is exactly the intent.

    recipe_dir and vocab_file read from the environment so the file stays
    free of machine-specific paths; with the variables unset they default to
    "." and resolve exactly as the 8-GPU smoke does, which is what lets this
    test diff the two.
    """
    monkeypatch.delenv("EMILIA_SMOKE_ROOT", raising=False)
    monkeypatch.delenv("EMILIA_RECIPE_ROOT", raising=False)

    smoke = _flatten(OmegaConf.to_container(OmegaConf.load(SMOKE_CONF), resolve=True))
    two = _flatten(
        OmegaConf.to_container(OmegaConf.load(SMOKE_2GPU_CONF), resolve=True)
    )

    assert set(smoke) == set(two), "2-GPU smoke adds/drops keys vs. the 8-GPU smoke"

    differing = {k for k in smoke if smoke[k] != two[k]}
    # Only two keys are changed by hand: num_device and exp_tag. Everything
    # else listed here is a DERIVED consequence via interpolation --
    # trainer.devices is ${num_device}; exp_dir/inference_dir and the logger's
    # name/id/save_dir/version all hang off ${exp_tag} or ${exp_dir}. Listing
    # them explicitly is what keeps this assertion tight: a genuine divergence
    # (batch_bins, accum, a sampler knob) still fails.
    expected = {
        "num_device",
        "trainer.devices",
        "exp_tag",
        "exp_dir",
        "inference_dir",
        "trainer.logger.name",
        "trainer.logger.id",
        "trainer.logger.save_dir",
        "trainer.logger.version",
    }
    assert differing <= expected, (
        f"2-GPU smoke diverges from the 8-GPU smoke in unexpected keys: "
        f"{sorted(differing - expected)}"
    )
    assert two["num_device"] == 2

    # The per-rank knobs that make the measurement transferable.
    for key in (
        "trainer.accumulate_grad_batches",
        "dataloader.train.iter_factory.batches.batch_bins",
        "dataloader.train.iter_factory.batches.min_batch_size",
        "dataloader.valid.iter_factory.batches.min_batch_size",
        "dataloader.train.iter_factory.batches.max_samples",
    ):
        assert two[key] == smoke[key], f"{key} must stay at the production value"

    # Instrument, not a training run: a quarter of the production global batch.
    bins = two["dataloader.train.iter_factory.batches.batch_bins"]
    accum = two["trainer.accumulate_grad_batches"]
    n_mels = two["n_mel_channels"]
    assert bins * accum * two["num_device"] / n_mels == 76800


def test_min_batch_size_is_one_on_both_loaders():
    """min_batch_size is a memory floor, not a topology knob.

    NumElementsArraySampler closes a batch on
    `current_count * current_length * n_mels > batch_bins` AND
    `current_count >= min_batch_size`, so any floor above 1 lets a batch of
    `floor` long utterances cost floor * L_max frames regardless of
    batch_bins. Emilia's 30s cap puts L_max at 2813 frames; at min_batch_size
    8 that peaked at 25,317 padded frames against a 4,800 nominal budget and
    OOM'd a V100-32 (job 43582509).

    Valid is asserted too, and deliberately: it shares train's batch_bins, so
    a floor left behind there reintroduces the identical batch. The first
    smoke survived that only because validation does not shuffle and
    limit_val_batches stopped before the long tail -- luck that evaporates
    the moment limit_val_batches is raised.
    """
    for conf_path in (CONF, SMOKE_CONF, SMOKE_2GPU_CONF):
        cfg = OmegaConf.load(conf_path)
        train = cfg.dataloader.train.iter_factory.batches
        valid = cfg.dataloader.valid.iter_factory.batches
        assert train.min_batch_size == 1, f"{conf_path.name}: train floor"
        assert valid.min_batch_size == 1, f"{conf_path.name}: valid floor"


def test_num_workers_is_inside_iter_factory():
    """num_workers/pin_memory must live INSIDE iter_factory or they do nothing.

    espnet3's DataLoaderBuilder._build_iter_factory instantiates only the
    `iter_factory` block, so keys placed as siblings of it are silently
    ignored -- no error, no warning, just single-process data loading. The
    2-GPU smoke (job 43635433) ran that way and spent iter_time=0.624s per
    step waiting on the loader against train_time=1.751s of compute, 26% of
    every step decoding mp3 on the main process.
    """
    for conf_path in (CONF, SMOKE_CONF, SMOKE_2GPU_CONF):
        cfg = OmegaConf.load(conf_path)
        for mode in ("train", "valid"):
            block = cfg.dataloader[mode]
            assert "num_workers" not in block, (
                f"{conf_path.name}: dataloader.{mode}.num_workers is a sibling "
                "of iter_factory, where espnet3 ignores it"
            )
            assert "pin_memory" not in block, (
                f"{conf_path.name}: dataloader.{mode}.pin_memory is a sibling "
                "of iter_factory, where espnet3 ignores it"
            )
            assert block.iter_factory.num_workers > 0, (
                f"{conf_path.name}: dataloader.{mode}.iter_factory.num_workers"
            )
            assert block.iter_factory.pin_memory is True
