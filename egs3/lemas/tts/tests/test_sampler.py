from src.sampler import BlockBatchSampler
from tests.test_dataset import _ds


def _blocks_of(ds, batch):
    return {ds.block_of(i) for i in batch} | {
        int(ds.cols.pack[i]) * 1000 for i in batch
    }


def test_batches_stay_inside_one_block_and_cover_every_row(corpus):
    # 2 s blocks: the 30 s de pack spans 15 blocks, the 12 s zh pack 6
    ds = _ds(corpus, block_samples=32000)
    ds.set_epoch(0)
    s = BlockBatchSampler(ds, batch_bins=100 * 300, n_mels=100, seed=1)  # 300 frames
    batches = list(s)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(ds)))
    for b in batches:
        assert len({(int(ds.cols.pack[i]), ds.block_of(i)) for i in b}) == 1
    cost = ds.frames_bound() * 100
    for b in batches:
        assert len(b) == 1 or cost[b].sum() <= 100 * 300


def test_epoch_reshuffles_and_validation_is_fixed_and_lengthwise(corpus):
    ds = _ds(corpus, block_samples=32000)
    ds.set_epoch(0)
    s = BlockBatchSampler(ds, batch_bins=100 * 300, seed=1)
    e0 = [list(b) for b in s]
    ds.set_epoch(1)
    e1 = [list(b) for b in s]
    assert len(s) == len(e1) and sorted(map(tuple, e0)) == sorted(map(tuple, e1))
    assert e0 != e1
    vd = _ds(corpus, split="valid", block_samples=32000)
    v = BlockBatchSampler(vd, batch_bins=100 * 3000, seed=1)
    vb = [list(b) for b in v]
    assert vb == [list(b) for b in v]  # fixed
    # validation ignores blocks: rows batched by length only
    assert any(len({(int(vd.cols.pack[i]), vd.block_of(i)) for i in b}) > 1 for b in vb)


def test_rank_sharding_partitions_the_batches(corpus):
    ds = _ds(corpus, block_samples=32000)
    ds.set_epoch(3)
    parts = [
        [
            tuple(b)
            for b in BlockBatchSampler(ds, 100 * 300, seed=1, rank=r, world_size=2)
        ]
        for r in range(2)
    ]
    assert len(parts[0]) == len(parts[1])
    assert not set(parts[0]) & set(parts[1])
    full = [
        tuple(b) for b in BlockBatchSampler(ds, 100 * 300, seed=1, rank=0, world_size=1)
    ]
    assert set(parts[0]) | set(parts[1]) <= set(full)


def test_frames_bound_is_mode_aware_upper_bound(corpus):
    ds = _ds(corpus)
    fb = ds.frames_bound()
    c = ds.cols
    lang_hi, spk_hi = ds.cfg["lang_prompt_sec"][1], ds.cfg["spk_prompt_sec"][1]
    for i in range(len(ds)):
        extra = lang_hi + (spk_hi if int(c.spk_mode[i]) == 1 else 0.0)
        assert fb[i] == 1 + int((float(c.dur[i]) + extra) * 24000) // 256
    ds.set_epoch(0)
    for i in range(len(ds)):
        assert len(ds[i]["speech"]) // 256 + 1 <= fb[i]
