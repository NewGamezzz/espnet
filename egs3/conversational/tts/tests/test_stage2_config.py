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
# 8 ch x 138 s (8 + 10 + 120 s assembled chunk sample at the Chorus
# chunk_window_max 120) probed by jobs/memcheck_stage2.py before launch;
# this pins the config's worst case to what that probe covers.
CHORUS_WINDOW_MAX = 120.0  # Thanapat 2026-08-28: cover all 4-8 speakers
WINDOW_MAX_DEFAULT = 80.0  # WindowParams default, used by the other corpora
CHORUS_SOLO_CEILING = 8 * (PROMPT_SLICE_MAX + PREV_SLICE_MAX + 120.0) * FS
# Stage-1 per-GPU budget per optimizer step: 7M bins x 7 accumulation.
STAGE1_ROWS_PER_STEP = 7_000_000 * 7


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


def test_chorus_long_windows_within_probed_ceiling():
    args = _chorus("train")["data_src_args"]
    # Chorus alone gets the long window range (train AND valid); the other
    # corpora keep the WindowParams defaults.
    for split in ("train", "valid"):
        wp = _chorus(split)["data_src_args"]["window_params"]
        assert wp == {"exclusion_mode": "cut", "window_max": CHORUS_WINDOW_MAX}, split
    for e in _entries("train"):
        if not e["data_src"].endswith("dataset_chorus"):
            assert "window_max" not in e["data_src_args"].get("window_params", {})
    ct = args["chunk_task"]
    assert ct["chunk_window_max"] == CHORUS_WINDOW_MAX
    assembled = PROMPT_SLICE_MAX + PREV_SLICE_MAX + ct["chunk_window_max"]
    worst = max(CHORUS_WINDOW_MAX, assembled)
    assert CHORUS_MAX_CHANNELS * worst * FS <= CHORUS_SOLO_CEILING
    assert CHORUS_SOLO_CEILING > CFG["dataloader"]["train"]["batch_bins"]
    assert CFG["model"]["arch"]["checkpoint_activations"] is True


def test_text_position_table_covers_the_longest_sample():
    # espnet2/tts/f5/backbones/dit.py raises when a sample exceeds this table.
    table = CFG["model"]["arch"]["text_precompute_max_pos"]
    longest = 0.0
    for e in _entries("train") + _entries("valid"):
        args = e["data_src_args"]
        window = args.get("window_params", {}).get("window_max", WINDOW_MAX_DEFAULT)
        ct = args.get("chunk_task")
        assembled = (
            PROMPT_SLICE_MAX + PREV_SLICE_MAX + ct["chunk_window_max"] if ct else 0.0
        )
        longest = max(longest, window, assembled)
    assert longest * FS / int(CFG["hop_length"]) <= table, (longest, table)


def test_chunk_task_knobs_everywhere_chunk_capable():
    seen = 0
    for e in _entries("train"):
        ct = e["data_src_args"].get("chunk_task")
        if e["data_src"].endswith("dataset_libritts"):
            assert ct is None
            continue
        seen += 1
        assert ct["chunk_task_prob"] == 0.7, e["data_src"]
        assert ct["prompt_only_prob"] == 0.5, e["data_src"]
        assert ct["prompt_slice_min"] == 2.0, e["data_src"]
        assert ct["prompt_speech_floor"] == 1.0, e["data_src"]
        if not e["data_src"].endswith("dataset_chorus"):
            assert ct["chunk_window_max"] == 60.0, e["data_src"]
    assert seen == 4


def test_batch_budget_per_optimizer_step_matches_stage1():
    bins = CFG["dataloader"]["train"]["batch_bins"]
    accum = CFG["trainer"]["accumulate_grad_batches"]
    assert bins == 9_800_000 and accum == 5
    assert bins * accum == STAGE1_ROWS_PER_STEP
    assert CFG["dataloader"]["valid"]["batch_bins"] == "${dataloader.train.batch_bins}"


def test_exclusion_cut_on_fisher_and_chorus_both_splits():
    for split in ("train", "valid"):
        for e in _entries(split):
            src = e["data_src"]
            mode = e["data_src_args"].get("window_params", {}).get("exclusion_mode")
            expect = (
                "cut" if src.endswith(("dataset_fisher", "dataset_chorus")) else None
            )
            assert mode == expect, (split, src)


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
    assert CFG["init_ckpt"] == "${recipe_dir}/init/backup_step98900.ckpt"
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
