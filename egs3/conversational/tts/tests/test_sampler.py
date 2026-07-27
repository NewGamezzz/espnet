"""ConversationBatchSampler: packing budget, epoch reshuffle, DDP alignment."""

from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from egs3.conversational.tts.src.lit_module import ConversationalLightningModule
from egs3.conversational.tts.src.sampler import (
    ConversationBatchSampler,
    pack_batches,
    window_costs,
)

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)

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


def test_set_epoch_matches_construction_epoch():
    dataset = fake_dataset([(10.0 + i, 2) for i in range(24)])
    kwargs = dict(batch_bins=2 * round(FS * 60.0), shuffle=True, seed=7)
    moved = ConversationBatchSampler(dataset, epoch=0, **kwargs)
    moved.set_epoch(3)
    built = ConversationBatchSampler(dataset, epoch=3, **kwargs)
    assert list(iter(moved)) == list(iter(built))
    moved.set_epoch(0)
    assert list(iter(moved)) == list(
        iter(ConversationBatchSampler(dataset, epoch=0, **kwargs))
    )


def test_sampler_alias_reaches_set_epoch_through_lightning():
    """Lightning only calls set_epoch on dataloader.sampler and
    dataloader.batch_sampler.sampler (lightning 2.6.5,
    fabric/utilities/data.py::_set_sampler_epoch), so the batch sampler
    must expose itself under .sampler."""
    import torch
    from lightning.fabric.utilities.data import _set_sampler_epoch

    dataset = fake_dataset([(10.0 + i, 2) for i in range(8)])
    sampler = ConversationBatchSampler(
        dataset, batch_bins=2 * round(FS * 60.0), shuffle=True, seed=0, epoch=0
    )
    assert sampler.sampler is sampler
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    _set_sampler_epoch(loader, 5)
    assert sampler.epoch == 5


# --------------------------------------------------------------------------
# Fit-loop-level ordering contract (Finding 1 in the final-review wave):
# lit_module.py's docstring explains that Lightning's FitLoop.run() eagerly
# materializes the first (and first-resumed) dataloader iterator via
# setup_data() *before* per-epoch set_epoch propagation runs, so the
# sampler's CONSTRUCTOR epoch - not a later set_epoch call - decides that
# iterator's batch order.  These tests pin that contract at the real
# lightning.pytorch.Trainer level (CPU, tiny synthetic data, no GPUs)
# instead of only unit-testing ConversationBatchSampler in isolation.
# --------------------------------------------------------------------------


class _RecordingBatchSampler(ConversationBatchSampler):
    """ConversationBatchSampler that records the epoch value seen by every
    __iter__ call, so a Trainer-level fit/resume exercises the same eager-
    iterator timing as production training."""

    def __init__(self, *args, epoch_log: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_log = epoch_log

    def __iter__(self):
        self.epoch_log.append(self.epoch)
        return super().__iter__()


class _TinyConversationDataset(torch.utils.data.Dataset):
    """Trivial dataset with the ``records``/``fs`` shape ConversationBatchSampler
    needs (see ``window_costs``), backing random tensors for a real
    LightningModule/Trainer fit - deliberately NOT ConversationDataset or the
    F5 model, which this test has no reason to drag in."""

    def __init__(self, n: int = 8, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 4, generator=gen)
        self.y = torch.randn(n, 1, generator=gen)
        self.records = [
            SimpleNamespace(t0=0.0, t1=1.0 + i, num_channels=1) for i in range(n)
        ]
        self.fs = 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class _ResumeOrderingModule(pl.LightningModule):
    """Minimal LightningModule mirroring ``_packed_dataloader``'s initial-epoch
    seeding in ``src/lit_module.py``: ``train_dataloader`` seeds the sampler
    from the fit loop's processed-epoch count when a trainer is attached
    (falling back to ``self.current_epoch`` beforehand), because that is the
    value in effect when ``FitLoop.run()`` eagerly materializes the first
    iterator - ``self.current_epoch`` (the completed-epoch count) is what a
    ``save_last`` checkpoint written from ``on_train_epoch_end`` stores one
    epoch behind, per the module docstring.

    ``use_processed_epoch=False`` reproduces the pre-fix behavior (seed from
    ``self.current_epoch``) so the regression this harness catches can be
    demonstrated directly, independent of lit_module.py's own fix.
    """

    def __init__(self, dataset, epoch_log: list, use_processed_epoch: bool = True):
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.dataset = dataset
        self.epoch_log = epoch_log
        self.use_processed_epoch = use_processed_epoch

    def _initial_epoch(self) -> int:
        if self.use_processed_epoch:
            # Delegate to the real production method (unbound: it only reads
            # self._trainer / self.current_epoch, both plain LightningModule
            # attributes) so this test fails if ConversationalLightningModule
            # regresses, instead of only checking a hand-rolled copy of it.
            return ConversationalLightningModule._initial_epoch(self)
        return self.current_epoch  # the pre-fix path, kept local on purpose

    def train_dataloader(self):
        sampler = _RecordingBatchSampler(
            self.dataset,
            batch_bins=10**9,  # everything fits in one batch; only order matters
            min_batch_size=1,
            shuffle=True,
            seed=0,
            epoch=self._initial_epoch(),
            epoch_log=self.epoch_log,
        )
        return torch.utils.data.DataLoader(self.dataset, batch_sampler=sampler)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _resume_ordering_trainer(tmp_path, max_epochs, ckpt_dir):
    """A Trainer configured like espnet3's recipe: loaders built once per fit
    (``reload_dataloaders_every_n_epochs=0``), no sanity pass, and a
    save-on-epoch-end ``save_last`` checkpoint mirroring
    ``espnet3.components.callbacks.default_callbacks.get_default_callbacks``."""
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        save_on_train_epoch_end=True,
        save_weights_only=False,
        filename="step{step}",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        limit_train_batches=1,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return trainer, checkpoint


def _run_fit_then_resume(tmp_path, use_processed_epoch: bool):
    """Fit 2 epochs, then resume for a 3rd from the ``save_last`` checkpoint.

    Returns ``(fresh_log, resumed_log)``: the epoch values ConversationBatchSampler
    saw at __iter__ time during each fit.
    """
    dataset = _TinyConversationDataset()
    ckpt_dir = tmp_path / "ckpt"

    fresh_log: list[int] = []
    trainer, checkpoint = _resume_ordering_trainer(
        tmp_path, max_epochs=2, ckpt_dir=ckpt_dir
    )
    module = _ResumeOrderingModule(dataset, fresh_log, use_processed_epoch)
    trainer.fit(module)

    resumed_log: list[int] = []
    trainer2, _ = _resume_ordering_trainer(tmp_path, max_epochs=3, ckpt_dir=ckpt_dir)
    module2 = _ResumeOrderingModule(dataset, resumed_log, use_processed_epoch)
    trainer2.fit(module2, ckpt_path=checkpoint.last_model_path)

    return fresh_log, resumed_log


def test_resume_first_epoch_uses_fit_loop_processed_epoch(tmp_path):
    """Finding 1 regression test: on resume, the first epoch must shuffle
    with a NEW epoch value, not replay the just-completed epoch's order.

    Seeding the sampler from ``trainer.fit_loop.epoch_progress.current.processed``
    (the fixed behavior) gives epochs 0, 1 on the fresh fit and epoch 2 on the
    resumed fit - never repeating 1, which is what pre-fix ``self.current_epoch``
    seeding (an espnet3 ``save_last`` checkpoint stores it one epoch behind)
    would produce (see the next test)."""
    fresh_log, resumed_log = _run_fit_then_resume(tmp_path, use_processed_epoch=True)
    assert fresh_log == [0, 1]
    assert resumed_log == [2]


def test_resume_first_epoch_replays_prior_order_pre_fix(tmp_path):
    """Same harness seeded from ``self.current_epoch`` (lit_module.py's
    pre-fix behavior): the resumed epoch incorrectly replays epoch 1's batch
    order instead of advancing to a fresh epoch 2 - this is the exact bug
    Finding 1 fixes, reproduced independently of lit_module.py's own code so
    the mechanism is pinned even if lit_module.py's implementation changes."""
    fresh_log, resumed_log = _run_fit_then_resume(tmp_path, use_processed_epoch=False)
    assert fresh_log == [0, 1]
    assert resumed_log == [1]  # the bug: replays the epoch-1 order, not epoch 2
