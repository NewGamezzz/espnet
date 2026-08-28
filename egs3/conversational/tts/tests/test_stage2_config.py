"""Stage-2 config invariants: memory cap, weights, coin, init, blocklists.

Reads conf/training_stage2_chorus_h100.yaml as plain YAML (no ${} resolution
is needed for these checks).
"""

from pathlib import Path

import yaml

RECIPE = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(
    (RECIPE / "conf/training_stage2_chorus_h100.yaml").read_text(encoding="utf-8")
)
FS = 24000
CHORUS_MAX_CHANNELS = 8  # measured on Delta 2026-08-27 (train: 7 meetings x 8)
# ChunkTaskParams defaults (dataset/preprocessing/chunk_task.py): the
# assembled chunk sample = prompt (<= prompt_slice_max) + prev (<= prev_slice_max)
# + chunk window.
PROMPT_SLICE_MAX = 8.0
PREV_SLICE_MAX = 10.0
# Solo-batch ceiling for Chorus under activation checkpointing (sample-rows):
# 8 ch x 80 s measured at 12.0 GiB on a GH200 (jobs/memcheck_stage2.py,
# 2026-08-28); this pins the config's worst case to what was measured.
CHORUS_SOLO_CEILING = 8 * 80.0 * FS
WINDOW_MAX_DEFAULT = 80.0  # WindowParams default; Chorus uses it uncapped


def _entries(split):
    return CFG["dataset"][split]


def _chorus(split):
    return next(e for e in _entries(split) if e["data_src"].endswith("dataset_chorus"))


def test_five_corpora_and_weights_sum_to_one():
    srcs = [e["data_src"] for e in _entries("train")]
    assert srcs == [
        "conversational/tts",
        "egs3.conversational.tts.dataset_libritts",
        "egs3.conversational.tts.dataset_candor",
        "egs3.conversational.tts.dataset_fisher",
        "egs3.conversational.tts.dataset_chorus",
    ]
    w = CFG["dataloader"]["train"]["weights"]
    assert w == [0.2, 0.1, 0.05, 0.35, 0.3]
    assert abs(sum(w) - 1.0) < 1e-9
    assert [e["data_src"] for e in _entries("valid")] == srcs


def test_chorus_uncapped_worst_case_matches_measured_ceiling():
    args = _chorus("train")["data_src_args"]
    assert "window_params" not in args  # same window range as the other corpora
    assert "window_params" not in _chorus("valid")["data_src_args"]
    ct = args["chunk_task"]
    assembled = PROMPT_SLICE_MAX + PREV_SLICE_MAX + ct["chunk_window_max"]
    worst = max(WINDOW_MAX_DEFAULT, assembled)
    assert CHORUS_MAX_CHANNELS * worst * FS <= CHORUS_SOLO_CEILING
    assert CHORUS_SOLO_CEILING > CFG["dataloader"]["train"]["batch_bins"]
    assert CFG["model"]["arch"]["checkpoint_activations"] is True


def test_mode_o_coin_everywhere():
    for e in _entries("train"):
        assert e["data_src_args"]["timestamp_align_prob"] == 0.2, e["data_src"]
    for e in _entries("valid"):
        assert "timestamp_align_prob" not in e["data_src_args"], e["data_src"]


def test_blocklists_on_candor_and_fisher_train_only():
    for e in _entries("train"):
        src = e["data_src"]
        has = "session_blocklist" in e["data_src_args"]
        assert has == (src.endswith("dataset_candor") or src.endswith("dataset_fisher"))
    for e in _entries("valid"):
        assert "session_blocklist" not in e["data_src_args"]


def test_init_keys_and_fresh_run_identity():
    assert CFG["init_ckpt"].endswith(".ckpt")
    assert CFG["model"]["init_ckpt"] == "${init_ckpt}"
    assert CFG["model"]["init_from_ema"] is True
    assert CFG["exp_tag"] == "train_stage2_chorus_h100"
    assert CFG["scheduler"]["total_steps"] == 100000
    assert CFG["trainer"]["max_steps"] == CFG["scheduler"]["total_steps"]
    # ~108 optimizer steps per epoch (smoke 2026-08-28): validate on an
    # epoch count, never a fraction of one.
    assert CFG["trainer"]["check_val_every_n_epoch"] == 10
    assert CFG["trainer"]["val_check_interval"] == 1.0
    assert CFG["fit"]["ckpt_path"] == "last"
    assert CFG["create_dataset"]["chorus_flac_dir"] == "${chorus_flac_root}"
