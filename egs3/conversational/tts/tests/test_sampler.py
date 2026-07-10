"""ConversationBatchSampler: packing budget, epoch reshuffle, DDP alignment."""

from types import SimpleNamespace

import pytest
from conftest import REPO_ROOT  # noqa: F401  (sys.path setup)

from egs3.conversational.tts.src.sampler import (
    ConversationBatchSampler,
    pack_batches,
    window_costs,
)

FS = 24000


def fake_dataset(windows):
    """``(duration_sec, num_channels)`` windows -> ConversationDataset stand-in."""
    records = [
        SimpleNamespace(t0=0.0, t1=duration, num_channels=n) for duration, n in windows
    ]
    return SimpleNamespace(records=records, fs=FS)


def test_window_costs_metadata_only():
    dataset = fake_dataset([(10.0, 2), (60.0, 3)])
    assert window_costs(dataset) == [(240000, 2), (1440000, 3)]


def test_window_costs_combined_dataset():
    combined = SimpleNamespace(
        datasets=[fake_dataset([(1.0, 2)]), fake_dataset([(2.0, 1)])]
    )
    assert window_costs(combined) == [(24000, 2), (48000, 1)]


def test_pack_batches_budget_and_coverage():
    windows = [(10.0 + i, 2) for i in range(10)] + [(60.0, 2), (5.0, 3)]
    costs = window_costs(fake_dataset(windows))
    bins = 2 * 3 * round(FS * 20.0)  # about three 20 s N=2 windows
    batches = pack_batches(costs, bins, min_batch_size=1)

    seen = sorted(i for batch in batches for i in batch)
    assert seen == list(range(len(windows)))  # exactly once each
    for batch in batches:
        rows = sum(costs[i][1] for i in batch)
        t_max = max(costs[i][0] for i in batch)
        assert rows * t_max <= bins or len(batch) == 1


def test_pack_batches_min_batch_size():
    costs = window_costs(fake_dataset([(30.0, 2)] * 7))
    batches = pack_batches(costs, batch_bins=1, min_batch_size=2)
    assert all(len(batch) >= 2 for batch in batches)
    assert sorted(i for b in batches for i in b) == list(range(7))


def test_epoch_reshuffle_is_seeded():
    dataset = fake_dataset([(10.0 + i, 2) for i in range(24)])
    make = lambda epoch: list(  # noqa: E731
        ConversationBatchSampler(
            dataset,
            batch_bins=2 * round(FS * 12.0),
            shuffle=True,
            seed=0,
            epoch=epoch,
        )
    )
    epoch0, epoch0_again, epoch1 = make(0), make(0), make(1)
    assert epoch0 == epoch0_again  # deterministic for a given epoch
    assert epoch0 != epoch1  # order reshuffles across epochs
    key = lambda batches: sorted(map(tuple, batches))  # noqa: E731
    assert key(epoch0) == key(epoch1)  # same composition, different order


def test_no_shuffle_is_stable():
    dataset = fake_dataset([(10.0 + i, 2) for i in range(8)])
    make = lambda epoch: list(  # noqa: E731
        ConversationBatchSampler(
            dataset, batch_bins=2 * round(FS * 12.0), shuffle=False, epoch=epoch
        )
    )
    assert make(0) == make(3)


def test_ddp_alignment(monkeypatch):
    dataset = fake_dataset([(10.0 + i, 2) for i in range(25)])

    def batches_for(rank, world_size):
        monkeypatch.setattr(
            ConversationBatchSampler,
            "_world_info",
            staticmethod(lambda: (rank, world_size)),
        )
        return list(
            ConversationBatchSampler(
                dataset,
                batch_bins=2 * round(FS * 11.0),  # one window per batch -> 25
                shuffle=True,
                seed=0,
                epoch=0,
            )
        )

    rank0, rank1 = batches_for(0, 2), batches_for(1, 2)
    assert len(rank0) == len(rank1) == 12  # 25 -> keep 24, split 12/12
    ids0 = {i for b in rank0 for i in b}
    ids1 = {i for b in rank1 for i in b}
    assert ids0.isdisjoint(ids1)
    assert len(ids0 | ids1) == 24  # one tail batch dropped


def test_ddp_too_few_batches_raises(monkeypatch):
    dataset = fake_dataset([(10.0, 2)])
    monkeypatch.setattr(
        ConversationBatchSampler, "_world_info", staticmethod(lambda: (0, 4))
    )
    sampler = ConversationBatchSampler(dataset, batch_bins=10**9)
    with pytest.raises(RuntimeError, match="world_size"):
        list(sampler)
