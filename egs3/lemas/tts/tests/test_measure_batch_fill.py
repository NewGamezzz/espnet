"""Batch fill: real frames the sampler's batches carry against their bound."""

import pytest
from dataset.dataset import LEMASDataset
from local.measure_batch_fill import batch_fill, measure
from omegaconf import OmegaConf
from src.sampler import BlockBatchSampler


def _ds(corpus, layout):
    return LEMASDataset(
        split="train",
        manifest_path=corpus["manifest"],
        token_list=corpus["tokens"],
        audio_root=corpus["audio"],
        prompt_config=dict(layout=layout, spk_prompt_sec=[1.0, 2.0]),
        seed=0,
    )


def _sampler(ds):
    return BlockBatchSampler(ds, batch_bins=100 * 2000, rank=0, world_size=1)


@pytest.mark.parametrize("layout", ["dualprompt", "stock"])
def test_fill_sums_the_drawn_frames_of_every_batch(corpus, layout):
    ds = _ds(corpus, layout)
    s = _sampler(ds)
    out = batch_fill(ds, s, epochs=[0, 1])
    real = bound = n = 0
    for epoch in (0, 1):
        ds.set_epoch(epoch)
        for rows in s:
            real += sum(ds.drawn_frames(i) for i in rows)
            bound += int(ds.frames_bound()[rows].sum())
            n += 1
    assert out["n_batches"] == n
    assert out["real_frames_per_batch"] == pytest.approx(real / n)
    assert out["bound_frames_per_batch"] == pytest.approx(bound / n)
    assert 0 < out["fill"] <= 1


def test_stock_batches_are_filled_to_their_bound(corpus):
    ds = _ds(corpus, "stock")
    assert batch_fill(ds, _sampler(ds), epochs=[0])["fill"] > 0.99


def test_measure_reads_the_train_set_and_sampler_from_the_config(corpus, tmp_path):
    cfg = dict(
        num_device=2,
        dataset=dict(
            train=[
                dict(
                    data_src_args=dict(
                        split="train",
                        manifest_path=str(corpus["manifest"]),
                        token_list=str(corpus["tokens"]),
                        audio_root=str(corpus["audio"]),
                        prompt_config=dict(layout="stock"),
                        seed=0,
                    )
                )
            ]
        ),
        batch_sampler=dict(
            _target_="src.sampler.BlockBatchSampler", batch_bins=200000, n_mels=100
        ),
        trainer=dict(accumulate_grad_batches=3),
    )
    path = tmp_path / "train.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    out = measure(str(path), epochs=1)
    ds = _ds(corpus, "stock")
    want = batch_fill(ds, _sampler(ds), epochs=[0])
    assert out["real_frames_per_batch"] == pytest.approx(want["real_frames_per_batch"])
    assert out["real_frames_per_update"] == pytest.approx(
        want["real_frames_per_batch"] * 3 * 2
    )
