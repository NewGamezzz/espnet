"""NumElementsArraySampler produces the same batches as the stock sampler."""

import numpy as np
import pytest

from egs3.emilia.tts.src.sampler import NumElementsArraySampler
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler


@pytest.fixture
def shape_file(tmp_path):
    rng = np.random.default_rng(0)
    lengths = rng.integers(28, 2812, size=5000)
    path = tmp_path / "feats_shape"
    with path.open("w", encoding="utf-8") as fh:
        for i, n in enumerate(lengths):
            fh.write(f"{i} {n},100\n")
    return path


def test_batches_are_identical_to_stock_sampler(shape_file):
    kwargs = dict(batch_bins=480000, shape_files=[str(shape_file)],
                  min_batch_size=8)
    stock = [tuple(sorted(b, key=int)) for b in NumElementsBatchSampler(**kwargs)]
    array = [tuple(sorted(map(str, b), key=int))
             for b in NumElementsArraySampler(**kwargs)]
    assert array == stock


def test_len_matches_stock(shape_file):
    kwargs = dict(batch_bins=480000, shape_files=[str(shape_file)],
                  min_batch_size=8)
    assert len(NumElementsArraySampler(**kwargs)) == len(
        NumElementsBatchSampler(**kwargs))


def test_max_samples_caps_short_utterance_batches(shape_file):
    """Upstream caps at 64; short utterances would otherwise give 300+."""
    sampler = NumElementsArraySampler(
        batch_bins=480000, shape_files=[str(shape_file)],
        min_batch_size=8, max_samples=64,
    )
    assert max(len(b) for b in sampler) <= 64


def test_every_index_appears_exactly_once(shape_file):
    sampler = NumElementsArraySampler(
        batch_bins=480000, shape_files=[str(shape_file)], min_batch_size=8)
    seen = [int(k) for batch in sampler for k in batch]
    assert sorted(seen) == list(range(5000))


def test_build_batch_sampler_registers_numel_array(shape_file):
    """espnet3's dataloader calls build_batch_sampler(**batches_config)
    directly (no Hydra _target_), so "numel_array" must resolve through
    espnet2/samplers/build_batch_sampler.py to this class, not just be
    constructible directly."""
    from espnet2.samplers.build_batch_sampler import build_batch_sampler

    sampler = build_batch_sampler(
        type="numel_array",
        batch_size=1,  # required by signature, unused for numel_array
        batch_bins=480000,
        shape_files=[str(shape_file)],
        min_batch_size=8,
        max_samples=64,
    )
    assert isinstance(sampler, NumElementsArraySampler)
    assert max(len(b) for b in sampler) <= 64


def test_max_samples_cap_holds_through_remainder_redistribution(tmp_path):
    """Regression for a gap the stock sampler's algorithm has no equivalent
    to guard against: NumElementsBatchSampler redistributes an undersized
    trailing remainder into other batches by checking only min_batch_size,
    which could push a batch over max_samples. Here max_samples=1 binds so
    tightly that every batch is already at the cap, so the redistribution
    step (batch_sizes[-1]=1 < min_batch_size=8 triggers it) must leave the
    leftover as its own batch rather than exceed the cap anywhere.
    Unreachable via this recipe's own configs, which set drop_last: true and
    so never populate a remainder to redistribute in the first place; this
    drives the code path directly since drop_last=False is still a
    supported constructor argument."""
    path = tmp_path / "feats_shape"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(f"{i} 100,100\n")

    sampler = NumElementsArraySampler(
        batch_bins=10**9,  # never closes a batch on bins
        shape_files=[str(path)],
        min_batch_size=8,
        max_samples=1,
        drop_last=False,
    )
    assert max(len(b) for b in sampler) <= 1
    seen = sorted(int(k) for b in sampler for k in b)
    assert seen == list(range(10))
