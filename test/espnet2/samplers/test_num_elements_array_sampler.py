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


@pytest.fixture
def multi_shape_files(tmp_path):
    """A second, independently-distributed shape file sharing the first's
    keys (e.g. a text-length file alongside the primary feats_shape file),
    to exercise the idx_map realignment path that a single-file fixture
    never reaches."""
    rng = np.random.default_rng(1)
    n = 600
    feats_lengths = rng.integers(28, 2812, size=n)
    text_lengths = rng.integers(1, 60, size=n)
    feats_path = tmp_path / "feats_shape"
    text_path = tmp_path / "text_shape"
    with feats_path.open("w", encoding="utf-8") as fh:
        for i, ln in enumerate(feats_lengths):
            fh.write(f"{i} {ln},100\n")
    with text_path.open("w", encoding="utf-8") as fh:
        for i, ln in enumerate(text_lengths):
            fh.write(f"{i} {ln},1\n")
    return [str(feats_path), str(text_path)]


def test_batches_are_identical_to_stock_sampler(shape_file):
    kwargs = dict(batch_bins=480000, shape_files=[str(shape_file)], min_batch_size=8)
    stock = [tuple(sorted(b, key=int)) for b in NumElementsBatchSampler(**kwargs)]
    array = [
        tuple(sorted(map(str, b), key=int)) for b in NumElementsArraySampler(**kwargs)
    ]
    assert array == stock


@pytest.mark.parametrize(
    "padding,drop_last,sort_in_batch,sort_batch,min_batch_size",
    [
        (True, False, "descending", "ascending", 8),
        (True, True, "descending", "ascending", 8),
        (True, False, "ascending", "descending", 4),
        (True, True, "ascending", "descending", 16),
        (False, False, "descending", "ascending", 8),
        (False, True, "ascending", "ascending", 1),
    ],
)
def test_batches_are_identical_to_stock_sampler_various_params(
    shape_file, padding, drop_last, sort_in_batch, sort_batch, min_batch_size
):
    """Exact (not just set-sorted) batch-for-batch equivalence across the
    padding/drop_last/sort_in_batch/sort_batch/min_batch_size combinations
    that the single fixed-default case in
    test_batches_are_identical_to_stock_sampler doesn't reach, including
    the non-padding running-sum branch (padding=False)."""
    kwargs = dict(
        batch_bins=480000,
        shape_files=[str(shape_file)],
        min_batch_size=min_batch_size,
        padding=padding,
        drop_last=drop_last,
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
    )
    stock = [tuple(b) for b in NumElementsBatchSampler(**kwargs)]
    array = [tuple(b) for b in NumElementsArraySampler(**kwargs)]
    assert array == stock


def test_batches_are_identical_to_stock_sampler_multi_file(multi_shape_files):
    """Exercises the idx_map realignment path (sampler.py's per-auxiliary-
    file reindexing into the primary file's sort order), which no
    single-shape-file test can reach."""
    kwargs = dict(batch_bins=50000, shape_files=multi_shape_files, min_batch_size=8)
    stock = [tuple(b) for b in NumElementsBatchSampler(**kwargs)]
    array = [tuple(b) for b in NumElementsArraySampler(**kwargs)]
    assert array == stock


def test_len_matches_stock(shape_file):
    kwargs = dict(batch_bins=480000, shape_files=[str(shape_file)], min_batch_size=8)
    assert len(NumElementsArraySampler(**kwargs)) == len(
        NumElementsBatchSampler(**kwargs)
    )


def test_max_samples_caps_short_utterance_batches(shape_file):
    """Upstream caps at 64; short utterances would otherwise give 300+."""
    sampler = NumElementsArraySampler(
        batch_bins=480000,
        shape_files=[str(shape_file)],
        min_batch_size=8,
        max_samples=64,
    )
    assert max(len(b) for b in sampler) <= 64


def test_every_index_appears_exactly_once(shape_file):
    sampler = NumElementsArraySampler(
        batch_bins=480000, shape_files=[str(shape_file)], min_batch_size=8
    )
    seen = [int(k) for batch in sampler for k in batch]
    assert sorted(seen) == list(range(5000))


def test_duplicate_key_raises(tmp_path):
    """A hand-rolled parser that silently kept both rows would emit the
    same utterance in two different batches, violating the "every index
    appears exactly once" invariant at runtime with no diagnostic. Stock's
    read_2columns_text raises on duplicates; this must too."""
    path = tmp_path / "feats_shape"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("0 100,100\n1 200,100\n0 150,100\n")

    with pytest.raises(RuntimeError, match="duplicate"):
        NumElementsArraySampler(
            batch_bins=480000, shape_files=[str(path)], min_batch_size=1
        )


def test_non_integer_key_raises_with_diagnostic(tmp_path):
    """The registration in build_batch_sampler.py is global (any espnet2
    recipe can select type: numel_array), so a recipe with string
    utterance ids must get an actionable error, not a bare ValueError from
    int()."""
    path = tmp_path / "feats_shape"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("uttA 100,100\n")

    with pytest.raises(RuntimeError, match="integer keys.*uttA"):
        NumElementsArraySampler(
            batch_bins=480000, shape_files=[str(path)], min_batch_size=1
        )


def test_max_samples_below_min_batch_size_raises(shape_file):
    """close_by_cap ignores min_batch_size by design (the cap is a hard
    ceiling); if max_samples < min_batch_size were allowed, every realized
    batch would end up under min_batch_size with no error until an 8-GPU
    run hit a DDP deadlock far downstream."""
    with pytest.raises(AssertionError):
        NumElementsArraySampler(
            batch_bins=480000,
            shape_files=[str(shape_file)],
            min_batch_size=8,
            max_samples=4,
        )


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
    which could push a batch over max_samples. max_samples must be >=
    min_batch_size (enforced by an assert), so the smallest legal pair that
    still forces the redistribution to stall is max_samples == min_batch_size
    == 8: with batch_bins effectively infinite, the cap alone packs the
    first 8 (of 10) same-length utterances into one batch, the remaining 2
    become an undersized trailing batch, and the redistribution step
    (batch_sizes[-1]=2 < min_batch_size=8 triggers it) finds the only other
    batch already at the cap and must leave the leftover as its own batch
    rather than exceed the cap. Unreachable via this recipe's own configs,
    which set drop_last: true and so never populate a remainder to
    redistribute in the first place; this drives the code path directly
    since drop_last=False is still a supported constructor argument."""
    path = tmp_path / "feats_shape"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(f"{i} 100,100\n")

    sampler = NumElementsArraySampler(
        batch_bins=10**9,  # never closes a batch on bins
        shape_files=[str(path)],
        min_batch_size=8,
        max_samples=8,
        drop_last=False,
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert sorted(len(b) for b in batches) == [2, 8]
    assert max(len(b) for b in batches) <= 8
    seen = sorted(int(k) for b in batches for k in b)
    assert seen == list(range(10))
