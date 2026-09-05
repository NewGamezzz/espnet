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
