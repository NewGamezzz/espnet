import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

TOKENS = ["<blank>", "<unk>", "a", "<spk>", "<lang>", "<de>", "<sos/eos>"]


def _tiny(tmp_path, path):
    cfg = OmegaConf.load(path)
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("\n".join(TOKENS) + "\n")
    cfg.token_list = str(tokens)
    cfg.model.hidden_size, cfg.model.depth, cfg.model.attention_heads = 32, 1, 2
    cfg.model.text_embedding_size, cfg.model.convolution_layers = 16, 1
    OmegaConf.resolve(cfg)
    return cfg


def test_training_config_instantiates_tiny_model(tmp_path):
    cfg = _tiny(tmp_path, "conf/training_f5_base_dualprompt.yaml")
    m = instantiate(cfg.model)
    assert type(m).__name__ == "DualPromptF5TTS"
    assert list(cfg.dataloader.collate_fn.not_sequence) == ["cond_frames"]
    assert list(cfg.create_shape.prompt_config.spk_prompt_sec) == [1.0, 6.0]
    assert cfg.dataset.train[0].data_src_args.prompt_config.p_drop_spk == 0.3
    assert cfg.dataset.train[0].data_src_args.prompt_config.p_drop_lang == 0.1
    assert cfg.create_token_list.token_type == "word"
    assert any(s.startswith("<lang>:") for s in cfg.create_token_list.add_symbol)


def test_base_config_geometry():
    cfg = OmegaConf.load("conf/training_f5_base_dualprompt.yaml")
    assert (cfg.model.hidden_size, cfg.model.depth, cfg.model.attention_heads) == (
        1024,
        22,
        16,
    )
    assert "collect_stats" not in cfg
    assert cfg.trainer.plugins[0]._target_.endswith("MmapCheckpointIO")


def test_smoke_config_differs_only_in_run_length(tmp_path):
    base = OmegaConf.load("conf/training_f5_base_dualprompt.yaml")
    smoke = OmegaConf.load("conf/training_smoke.yaml")
    assert smoke.model == base.model and smoke.dataloader == base.dataloader
    assert smoke.trainer.max_steps < base.trainer.max_steps
    assert smoke.exp_tag != base.exp_tag
    m = instantiate(_tiny(tmp_path, "conf/training_smoke.yaml").model)
    assert type(m).__name__ == "DualPromptF5TTS"


BASE = "conf/training_f5_base_dualprompt.yaml"
SMOKES = ["conf/training_smoke.yaml", "conf/training_smoke_1gpu.yaml"]


@pytest.mark.parametrize("path", [BASE] + SMOKES)
def test_epochs_are_minutes_long_and_end_on_an_optimizer_step(path):
    # espnet3 writes last.ckpt at the end of an epoch and nowhere else, and a
    # chain link is killed after 1 h: an epoch of one pass (154 h) left the
    # first production run without any checkpoint to resume from.
    cfg = OmegaConf.load(path)
    per_epoch = cfg.batch_sampler.batches_per_epoch
    assert per_epoch % cfg.trainer.accumulate_grad_batches == 0
    assert per_epoch / 3.3 <= 10 * 60  # 3.3 micro-batches per second on 4 A100
    assert cfg.trainer.max_epochs == -1  # the run is bounded by max_steps
    # an integer val_check_interval counts micro-batches and validates mid-epoch
    assert "val_check_interval" not in cfg.trainer
    assert cfg.trainer.check_val_every_n_epoch >= 1


@pytest.mark.parametrize("path", SMOKES)
def test_smoke_batches_as_production_does(path):
    base = OmegaConf.to_container(OmegaConf.load(BASE).batch_sampler)
    smoke = OmegaConf.to_container(OmegaConf.load(path).batch_sampler)
    assert smoke.pop("batches_per_epoch") < base.pop("batches_per_epoch")
    assert smoke == base


def test_backups_are_written_at_epoch_ends_every_25k_steps():
    # only a checkpoint written at the end of an epoch resumes exactly
    cfg = OmegaConf.load(BASE)
    (backup,) = [
        c for c in cfg.trainer.callbacks if c._target_.endswith("ModelCheckpoint")
    ]
    steps = cfg.batch_sampler.batches_per_epoch // cfg.trainer.accumulate_grad_batches
    assert backup.save_on_train_epoch_end and "every_n_train_steps" not in backup
    assert backup.every_n_epochs * steps == 25000 and backup.save_top_k == -1
