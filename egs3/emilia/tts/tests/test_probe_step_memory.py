"""local/probe_step_memory.py: per-GPU memory of a full training step.

The probe picks ``batch_bins`` for a GPU type from a measurement instead of a
guess. Its batch shapes must be the ones the production sampler really
produces, so the replay is pinned against espnet2's own sampler here.
"""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from egs3.emilia.tts.local.probe_step_memory import (
    load_frame_lengths,
    peak_batches,
    replay_numel_batches,
    run_probe,
)
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler

RECIPE_ROOT = Path(__file__).resolve().parents[1]
N_MELS = 100


def _write_shape_file(path: Path, lengths) -> Path:
    path.write_text(
        "".join(f"{i} {int(n)},{N_MELS}\n" for i, n in enumerate(lengths)),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def lengths():
    """Emilia-like frame counts: 3-30 s at 24 kHz / hop 256, unsorted."""
    rng = np.random.default_rng(0)
    return rng.integers(282, 2814, size=3000)


@pytest.mark.parametrize("batch_bins", [480_000, 960_000, 2_000_000])
@pytest.mark.parametrize("drop_last", [True, False])
def test_replay_matches_the_production_sampler(
    tmp_path, lengths, batch_bins, drop_last
):
    shape_file = _write_shape_file(tmp_path / "feats_shape", lengths)
    sampler = NumElementsBatchSampler(
        batch_bins=batch_bins,
        shape_files=[str(shape_file)],
        min_batch_size=1,
        drop_last=drop_last,
    )
    expected = [
        (len(batch), max(int(lengths[int(k)]) for k in batch)) for batch in sampler
    ]

    sizes, max_frames = replay_numel_batches(
        load_frame_lengths(shape_file), N_MELS, batch_bins, drop_last=drop_last
    )

    assert list(zip(sizes.tolist(), max_frames.tolist())) == expected


def test_every_closed_batch_exceeds_the_nominal_budget(lengths):
    """batch_bins / n_mels is a floor on padded frames, not a ceiling.

    The sampler closes a batch on the first count that EXCEEDS the budget, so
    memory has to be measured at the realized shapes, never at the nominal.
    """
    sizes, max_frames = replay_numel_batches(
        np.asarray(lengths), N_MELS, 960_000, drop_last=True
    )
    assert (sizes * max_frames > 9600).all()


def test_peak_batches_cover_the_three_ways_a_batch_can_be_the_worst():
    sizes = np.array([30, 12, 5, 4])
    max_frames = np.array([300, 900, 2000, 2800])
    peaks = peak_batches(sizes, max_frames)
    assert peaks == {
        "most_utterances": (30, 300),
        "most_padded_frames": (4, 2800),
        "longest_utterance": (4, 2800),
    }


def test_peak_batches_breaks_padded_frame_ties_towards_the_longer_batch():
    """Attention cost grows with length, so the longer of two ties is worse."""
    sizes = np.array([20, 10])
    max_frames = np.array([500, 1000])
    assert peak_batches(sizes, max_frames)["most_padded_frames"] == (10, 1000)


def test_load_frame_lengths_reads_the_recipe_shape_format(tmp_path):
    shape_file = _write_shape_file(tmp_path / "feats_shape", [300, 2813, 1000])
    loaded = load_frame_lengths(shape_file)
    assert loaded.tolist() == [300, 2813, 1000]


@pytest.fixture
def tiny_config(tmp_path):
    """The production config with the transformer shrunk to run on a CPU."""
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("".join(f"tok{i}\n" for i in range(50)), encoding="utf-8")
    cfg = OmegaConf.load(RECIPE_ROOT / "conf" / "training_f5_tts_base.yaml")
    cfg.recipe_dir = str(RECIPE_ROOT)
    cfg.vocab_file = str(vocab)
    cfg.model.tts_conf.update(
        dim=32, depth=1, heads=2, dim_head=16, text_dim=16, conv_layers=1
    )
    return cfg


def test_run_probe_reports_one_row_per_precision_and_batch_shape(tiny_config):
    rows = run_probe(
        tiny_config,
        shapes={"a": (2, 40), "b": (1, 60)},
        precisions=["32-true", "bf16-mixed"],
        device="cpu",
        steps=1,
    )
    assert [
        (r["precision"], r["batch"], r["utterances"], r["frames"]) for r in rows
    ] == [
        ("32-true", "a", 2, 40),
        ("32-true", "b", 1, 60),
        ("bf16-mixed", "a", 2, 40),
        ("bf16-mixed", "b", 1, 60),
    ]
    for row in rows:
        assert row["status"] == "ok"
        assert np.isfinite(row["loss"])
        assert row["seconds_per_step"] > 0


def test_run_probe_trains_the_model_it_measures(tiny_config):
    """A step that does not update the weights measures the wrong memory."""
    import torch

    seen = {}

    def spy(model):
        seen["model"] = model
        seen["before"] = [p.detach().clone() for p in model.parameters()]

    run_probe(
        tiny_config,
        shapes={"a": (2, 40)},
        precisions=["32-true"],
        device="cpu",
        steps=1,
        on_model_built=spy,
    )
    after = list(seen["model"].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(seen["before"], after))


def test_run_probe_frees_a_failed_step_before_the_next_measurement(tiny_config):
    """An out-of-memory step must not leak into the rows measured after it.

    On the A100 the loss of the first failed step stayed referenced, its
    autograd graph kept the GPU full, and every later row (all of bf16)
    reported "oom" whatever its size.
    """
    import weakref

    import torch

    failed_loss_alive = []

    def spy(model):
        real_forward = model.forward
        state = {"calls": 0, "failed_loss": None}

        def forward(**batch):
            state["calls"] += 1
            if state["failed_loss"] is not None:
                failed_loss_alive.append(state["failed_loss"]() is not None)
                state["failed_loss"] = None
            loss, stats, weight = real_forward(**batch)
            if state["calls"] == 1:

                def out_of_memory(grad):
                    raise torch.OutOfMemoryError("simulated")

                loss.register_hook(out_of_memory)
                state["failed_loss"] = weakref.ref(loss)
            return loss, stats, weight

        model.forward = forward

    rows = run_probe(
        tiny_config,
        shapes={"a": (2, 40), "b": (1, 60)},
        precisions=["32-true"],
        device="cpu",
        steps=1,
        on_model_built=spy,
    )

    assert [row["status"] for row in rows] == ["oom", "ok"]
    assert failed_loss_alive == [False]


def test_run_probe_rejects_an_unknown_precision(tiny_config):
    with pytest.raises(ValueError, match="precision"):
        run_probe(
            tiny_config,
            shapes={"a": (1, 40)},
            precisions=["16-mixed"],
            device="cpu",
            steps=1,
        )
