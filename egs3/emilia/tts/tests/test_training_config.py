"""The Base training config matches the paper and the spec's decisions."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONF = Path(__file__).resolve().parents[1] / "conf" / "training_f5_tts_base.yaml"


@pytest.fixture
def cfg():
    return OmegaConf.load(CONF)


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
    """Spec 6.2: 12 epochs total, so per-epoch validation is useless."""
    assert "check_val_every_n_epoch" not in cfg.trainer
    assert cfg.trainer.val_check_interval > 0
    assert cfg.trainer.limit_val_batches > 0


def test_uses_f5_pinyin_preprocessor_not_common(cfg):
    assert cfg.dataset.preprocessor._target_ == (
        "espnet2.text.f5_preprocessor.F5PinyinPreprocessor"
    )


def test_normalize_is_null_and_no_collect_stats(cfg):
    assert cfg.model.normalize is None
    assert "collect_stats" not in cfg
    assert "remove_long_short" not in cfg
    assert "create_token_list" not in cfg


def test_auto_resume_is_configured(cfg):
    """Preemption recovery on PSC depends on this."""
    assert str(cfg.fit.ckpt_path).endswith("last.ckpt")


def test_recipe_dir_is_set(cfg):
    """create_shape reads training_config.recipe_dir by plain attribute
    access with no fallback (src/system.py), so this pins the contract."""
    assert cfg.recipe_dir is not None

    # Check raw (unresolved) interpolation strings, not resolved values:
    # data_dir/exp_dir must reference ${recipe_dir} directly, and stats_dir
    # must chain through exp_dir back to it.
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert "${recipe_dir}" in raw["data_dir"]
    assert "${recipe_dir}" in raw["exp_dir"]
    assert "${exp_dir}" in raw["stats_dir"]


def test_create_shape_block_has_manifest_paths(cfg):
    """src/system.py:100 does cfg["manifest_paths"][split] and would
    KeyError without both splits present."""
    manifest_paths = cfg.create_shape.manifest_paths
    assert "train" in manifest_paths
    assert "valid" in manifest_paths
