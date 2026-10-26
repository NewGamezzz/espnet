from pathlib import Path

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
    assert list(cfg.prompt_config.spk_prompt_sec) == [1.0, 6.0]
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


def test_base_config_is_the_delta_low_lr_run_on_600k_steps_in_bf16():
    # Thanapat 2026-10-24: relaunch the 7.5e-5 run on Delta with the 600k
    # schedule and bf16 mixed precision, under a new run name
    cfg = OmegaConf.load(BASE)
    assert cfg.exp_tag == "train_f5_base_lemas_dualprompt_bf16_r3"
    assert cfg.optimizer.lr == 7.5e-5 and cfg.num_device == 4
    assert cfg.trainer.max_steps == cfg.scheduler.total_steps == 600_000
    assert cfg.trainer.precision == "bf16-mixed"


def test_smoke_config_differs_only_in_run_length(tmp_path):
    base = OmegaConf.load("conf/training_f5_base_dualprompt.yaml")
    smoke = OmegaConf.load("conf/training_smoke.yaml")
    assert smoke.model == base.model and smoke.dataloader == base.dataloader
    assert smoke.trainer.max_steps < base.trainer.max_steps
    assert smoke.exp_tag != base.exp_tag
    m = instantiate(_tiny(tmp_path, "conf/training_smoke.yaml").model)
    assert type(m).__name__ == "DualPromptF5TTS"


BASE = "conf/training_f5_base_dualprompt.yaml"
GH200 = "conf/training_f5_base_dualprompt_gh200.yaml"
DIAG_A = "conf/training_f5_small_en_dualprompt.yaml"
DIAG_B = "conf/training_f5_small_en_stock.yaml"
# Real frames per micro-batch, measured by local/measure_batch_fill.py over 3
# epochs (Delta job 23156524, 2026-10-24). The sampler sizes batches on a
# bound with both prompts at their longest, and a batch never leaves its 4 MB
# pack block, so dual-prompt batches are 49-70% full and B's (no prompts) 100%.
MEASURED_REAL_FRAMES = {
    BASE: 3261.45,
    GH200: 7512.07,
    DIAG_A: 7408.57,
    DIAG_B: 7924.93,
    # the baseline rows, Delta job 23217789 (2026-10-26; BASE re-read 3274.45)
    "conf/training_f5_base_s0c_stock_phones.yaml": 2842.61,
    "conf/training_f5_base_s0a_stock_chars.yaml": 2839.61,
    "conf/training_f5_base_d1_spkonly.yaml": 3640.04,
}
SMOKES = ["conf/training_smoke.yaml", "conf/training_smoke_1gpu.yaml"]


@pytest.mark.parametrize(
    "path", [BASE, "conf/training_f5_base_dualprompt_gh200.yaml"] + SMOKES
)
def test_epochs_are_minutes_long_and_end_on_an_optimizer_step(path):
    # espnet3 writes last.ckpt at the end of an epoch and nowhere else, and a
    # chain link is killed after 1 h: an epoch of one pass (154 h) left the
    # first production run without any checkpoint to resume from.
    cfg = OmegaConf.load(path)
    per_epoch = cfg.batch_sampler.batches_per_epoch
    assert per_epoch % cfg.trainer.accumulate_grad_batches == 0
    # 3.3 micro-batches per second on 4 A100 (fp32, r2): at F5's batch an
    # epoch of 100 steps is under 12 min, the most a killed 1 h link loses
    assert per_epoch / 3.3 <= 12 * 60
    assert cfg.trainer.max_epochs == -1  # the run is bounded by max_steps
    # an integer val_check_interval counts micro-batches and validates mid-epoch
    assert "val_check_interval" not in cfg.trainer
    # the resume is exact only with a validation at every epoch end
    assert cfg.trainer.check_val_every_n_epoch == 1


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


def test_gh200_config_differs_from_base_only_in_the_intended_knobs():
    # DeltaAI run (Thanapat 2026-10-01): 2 GPUs, 2x learning rate, and the
    # GH200's memory spent on bigger micro-batches; both at F5's batch
    base = OmegaConf.to_container(OmegaConf.load(BASE))
    gh = OmegaConf.to_container(OmegaConf.load(GH200))
    assert gh["exp_tag"] != base["exp_tag"]
    assert gh["num_device"] == 2 and base["num_device"] == 4
    assert gh["optimizer"]["lr"] == 1.5e-4 and base["optimizer"]["lr"] == 7.5e-5
    assert gh["trainer"]["precision"] == base["trainer"]["precision"] == "bf16-mixed"
    # Thanapat 2026-10-24: both runs at F5's batch (307,200 frames per update)
    # in REAL frames. The sampler's bound assumes both prompts at their
    # longest, so micro-batches are about half full (measured below).
    real = {
        c: MEASURED_REAL_FRAMES[c]
        * x["num_device"]
        * x["trainer"]["accumulate_grad_batches"]
        for c, x in ((BASE, base), (GH200, gh))
    }
    assert base["trainer"]["accumulate_grad_batches"] == 23
    assert gh["trainer"]["accumulate_grad_batches"] == 20
    for c in real:
        assert abs(real[c] / 307_200 - 1) < 0.03, (c, real[c])
    steps = lambda c: (  # noqa: E731
        c["batch_sampler"]["batches_per_epoch"]
        // c["trainer"]["accumulate_grad_batches"]
    )
    assert steps(gh) == steps(base) == 100
    assert gh["trainer"]["max_steps"] == gh["scheduler"]["total_steps"] == 600_000
    for c in (base, gh):
        c["exp_tag"] = c["optimizer"]["lr"] = c["num_device"] = None
        c["batch_sampler"]["batch_bins"] = c["batch_sampler"]["batches_per_epoch"] = (
            None
        )
        c["trainer"]["accumulate_grad_batches"] = None
    assert gh == base


SMALL = dict(
    hidden_size=768,
    depth=18,
    attention_heads=12,
    attention_head_size=64,
    feed_forward_multiplier=2,
    text_embedding_size=512,
    convolution_layers=4,
)


def _backup(c):
    return next(
        cb
        for cb in c["trainer"]["callbacks"]
        if cb.get("filename") == "backup_step{step}"
    )


def test_layout_diagnostic_a_is_the_gh200_run_shrunk_to_english_on_one_gpu():
    # Thanapat 2026-10-23: F5-Small, English only, 1 A100, bf16; the step axis
    # (lr, schedule, 320k frames per update) stays the GH200 run's
    gh = OmegaConf.to_container(OmegaConf.load(GH200))
    a = OmegaConf.to_container(OmegaConf.load(DIAG_A))
    assert a["num_device"] == 1 and a["trainer"]["precision"] == "bf16-mixed"
    assert {k: a["model"][k] for k in SMALL} == SMALL
    # batch geometry is A's own: it matches B's real frames per update
    # (test_layout_diagnostic_b_differs_from_a_only_in_the_layout)
    steps = (
        a["batch_sampler"]["batches_per_epoch"]
        // a["trainer"]["accumulate_grad_batches"]
    )
    assert steps == 100
    assert _backup(a)["every_n_epochs"] * steps == 5_000
    for split in ("train", "valid"):
        args = a["dataset"][split][0]["data_src_args"]
        assert args["manifest_path"] == "${data_dir}/manifest_en/" + split + ".tsv"
    assert a["stats_dir"] == "${recipe_dir}/exp/stats_en_dualprompt"
    assert a["prompt_config"].pop("layout") == "dualprompt"  # the default
    for c in (gh, a):
        c["exp_tag"] = c["num_device"] = c["stats_dir"] = None
        for k in SMALL:
            c["model"][k] = None
        c["batch_sampler"]["batch_bins"] = c["batch_sampler"]["batches_per_epoch"] = (
            None
        )
        c["trainer"]["accumulate_grad_batches"] = None
        for split in ("train", "valid"):
            c["dataset"][split][0]["data_src_args"]["manifest_path"] = None
        _backup(c)["every_n_epochs"] = None
    assert a == gh


def test_layout_diagnostic_b_differs_from_a_only_in_the_layout():
    a = OmegaConf.to_container(OmegaConf.load(DIAG_A))
    b = OmegaConf.to_container(OmegaConf.load(DIAG_B))
    assert a["prompt_config"]["layout"] == "dualprompt"
    assert b["prompt_config"]["layout"] == "stock"
    assert b["stats_dir"] == "${recipe_dir}/exp/stats_en_stock"
    assert a["exp_tag"] != b["exp_tag"]
    # Thanapat 2026-10-24: equal REAL frames per update. A keeps B's
    # micro-batch (no new memory risk) and accumulates more of them; 21 is
    # the closest integer (-1.8%; 22 would be +2.8%). Epochs stay 100 steps.
    acc = {
        DIAG_A: a["trainer"]["accumulate_grad_batches"],
        DIAG_B: b["trainer"]["accumulate_grad_batches"],
    }
    assert acc == {DIAG_A: 21, DIAG_B: 20}
    real = {c: MEASURED_REAL_FRAMES[c] * acc[c] for c in acc}
    assert abs(real[DIAG_A] / real[DIAG_B] - 1) < 0.02
    for c in (a, b):
        assert (
            c["batch_sampler"]["batches_per_epoch"]
            == 100 * c["trainer"]["accumulate_grad_batches"]
        )
        c["exp_tag"] = c["stats_dir"] = c["prompt_config"]["layout"] = None
        c["trainer"]["accumulate_grad_batches"] = None
        c["batch_sampler"]["batches_per_epoch"] = None
    assert a == b


@pytest.mark.parametrize("path", [DIAG_A, DIAG_B])
def test_layout_diagnostics_instantiate(tmp_path, path):
    cfg = _tiny(tmp_path, path)
    assert type(instantiate(cfg.model)).__name__ == "DualPromptF5TTS"


@pytest.mark.parametrize(
    "path", sorted(str(p) for p in Path("conf").glob("training_*.yaml"))
)
def test_configs_carry_no_block_for_a_stage_run_py_does_not_define(path):
    # a removed stage (create_shape, 2026-09) left dead blocks in every config
    import run
    from src.system import TTSSystem

    cfg = OmegaConf.load(path)
    stages = set(run.DEFAULT_STAGES)
    stray = [
        k for k in cfg if callable(getattr(TTSSystem, k, None)) and k not in stages
    ]
    assert stray == [], stray
    assert not hasattr(TTSSystem, "create_shape")


# ---- paper baseline rows at F5-Base on all 10 languages (design note section 14,
# Thanapat 2026-10-24) -------------------------------------------------------
S0C = "conf/training_f5_base_s0c_stock_phones.yaml"
S0A = "conf/training_f5_base_s0a_stock_chars.yaml"
D1 = "conf/training_f5_base_d1_spkonly.yaml"
# keys each row changes against BASE (the Delta D2 run); everything else is
# pinned equal. lr 1.5e-4 pairs the rows with the GH200 D2 run.
ROW_DELTAS = {
    S0C: {
        "exp_tag": "train_f5_base_lemas_s0c_stock_phones",
        "stats_dir": "${recipe_dir}/exp/stats_s0c",
        ("prompt_config", "layout"): "stock",
        ("batch_sampler", "batch_bins"): 400_000,
    },
    S0A: {
        "exp_tag": "train_f5_base_lemas_s0a_stock_chars",
        "stats_dir": "${recipe_dir}/exp/stats_s0a",
        "data_dir": "${recipe_dir}/data_chars",
        ("prompt_config", "layout"): "stock",
        ("prompt_config", "text_tag"): False,
        ("create_dataset", "text_units"): "chars",
        ("create_dataset", "data_path"): "data_chars",
        ("batch_sampler", "batch_bins"): 400_000,
    },
    D1: {
        "exp_tag": "train_f5_base_lemas_d1_spkonly",
        "stats_dir": "${recipe_dir}/exp/stats_d1",
        ("prompt_config", "p_drop_lang"): 1.0,
        # 800k went OOM on rank 0 (smoke 23218469: 36 GB allocated at epoch 2;
        # without the language prompt a micro-batch holds 7.6 rows against
        # D2's 4.7, so it pads to more frames at the same bound)
        ("batch_sampler", "batch_bins"): 400_000,
    },
}


def _get(c, key):
    return c[key] if isinstance(key, str) else c[key[0]][key[1]]


def _set(c, key, value):
    if isinstance(key, str):
        c[key] = value
    else:
        c.setdefault(key[0], {})[key[1]] = value


@pytest.mark.parametrize("path", [S0C, S0A, D1])
def test_baseline_row_differs_from_base_only_in_its_knobs(path):
    base = OmegaConf.to_container(OmegaConf.load(BASE))
    row = OmegaConf.to_container(OmegaConf.load(path))
    gh = OmegaConf.to_container(OmegaConf.load(GH200))
    for key, value in ROW_DELTAS[path].items():
        assert _get(row, key) == value, key
    assert row["optimizer"]["lr"] == gh["optimizer"]["lr"] == 1.5e-4
    # 100 optimizer steps per epoch; accumulate is set from the measured fill
    acc = row["trainer"]["accumulate_grad_batches"]
    assert row["batch_sampler"]["batches_per_epoch"] == 100 * acc
    for c in (base, row):
        for key in ROW_DELTAS[path]:
            _set(c, key, None)
        c["optimizer"]["lr"] = None
        c["trainer"]["accumulate_grad_batches"] = None
        c["batch_sampler"]["batches_per_epoch"] = None
    assert row == base


@pytest.mark.parametrize("path", [S0C, S0A, D1])
def test_baseline_rows_instantiate(tmp_path, path):
    cfg = _tiny(tmp_path, path)
    assert type(instantiate(cfg.model)).__name__ == "DualPromptF5TTS"


def test_stock_rows_keep_their_own_data_and_token_list():
    # raw strings (no interpolation): every data path of S0a hangs off data_dir,
    # so the one data_dir change moves manifests, token list and lang_stats
    s0a = OmegaConf.to_container(OmegaConf.load(S0A), resolve=False)
    s0c = OmegaConf.to_container(OmegaConf.load(S0C), resolve=False)
    for split in ("train", "valid"):
        args = s0a["dataset"][split][0]["data_src_args"]
        assert args["manifest_path"] == "${data_dir}/manifest/" + split + ".tsv"
    assert s0a["token_list"] == "${data_dir}/tokens/tokens.txt"
    assert s0a["create_token_list"]["manifest_path"] == "${data_dir}/manifest/train.tsv"
    assert s0a["create_token_list"]["save_path"] == "${data_dir}/tokens"
    resolved = OmegaConf.load(S0A)
    assert resolved.token_list == "./data_chars/tokens/tokens.txt"
    assert s0c["data_dir"] == "${recipe_dir}/data"
    assert "text_tag" not in s0c["prompt_config"]  # S0c keeps the tag (default)


@pytest.mark.parametrize("path", [S0C, S0A, D1])
def test_baseline_rows_match_the_frame_budget(path):
    # every row sees the production budget of 307,200 real frames per
    # optimizer step (D2: 4 GPU x measured fill x accumulate); the stock rows
    # fill their 400k bound (fill 1.00) and D1 fills 0.56 of its 800k bound
    # without the language prompt's cap, both from local/measure_batch_fill.py
    row = OmegaConf.to_container(OmegaConf.load(path))
    acc = row["trainer"]["accumulate_grad_batches"]
    real = MEASURED_REAL_FRAMES[path] * row["num_device"] * acc
    assert abs(real / 307_200 - 1) < 0.01, (path, real)
    assert acc == {S0C: 27, S0A: 27, D1: 21}[path]
