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
# Solo-batch ceiling for Chorus under activation checkpointing (sample-rows);
# the smoke run measures the real peak, this pins the config's intent.
CHORUS_SOLO_CEILING = 8 * 60.0 * FS


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


def test_chorus_caps_fit_solo_ceiling_with_checkpointing():
    args = _chorus("train")["data_src_args"]
    window_max = args["window_params"]["window_max"]
    ct = args["chunk_task"]
    assembled = PROMPT_SLICE_MAX + PREV_SLICE_MAX + ct["chunk_window_max"]
    worst = max(window_max, assembled)
    assert CHORUS_MAX_CHANNELS * worst * FS <= CHORUS_SOLO_CEILING
    assert (
        _chorus("valid")["data_src_args"]["window_params"]["window_max"] == window_max
    )
    # The ceiling exceeds batch_bins only because activations are checkpointed.
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
    assert CFG["fit"]["ckpt_path"] == "last"
    assert CFG["create_dataset"]["chorus_flac_dir"] == "${chorus_flac_root}"
