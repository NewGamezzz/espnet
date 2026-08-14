"""ConversationBatchSampler: packing budget, epoch reshuffle, DDP alignment."""

import logging
import time
from types import SimpleNamespace

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from egs3.conversational.tts.dataset.preprocessing.planner import (
    WindowParams,
    plan_sessions,
)
from egs3.conversational.tts.dataset.preprocessing.sessions import (
    SessionRecord,
    write_session_manifest,
)
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.dataset import ConversationDataset
from egs3.conversational.tts.src.lit_module import ConversationalLightningModule
from egs3.conversational.tts.src.sampler import (
    ConversationBatchSampler,
    pack_batches,
    record_costs,
)

from .conftest import REPO_ROOT  # noqa: F401  (sys.path setup)

FS = 24000
# Generous enough to hold several of the ~4-10s windows below per batch,
# small enough that mixed_dataset's ~60/40s sessions still pack into several
# batches (not one giant batch) - exact counts are not asserted, only budget
# and composition invariants.
BB = 3 * round(FS * 8.0)


def fake_dataset(windows, *, session_id="s", fs=FS):
    """``(duration_sec, num_channels)`` windows -> ConversationDataset stand-in.

    Records are plain ``SimpleNamespace`` objects carrying ``window_id``/
    ``session_id`` (for spec-yield identity assertions below) plus the
    ``t0``/``t1``/``num_channels`` fields the sampler's cost model reads.
    """
    records = [
        SimpleNamespace(
            window_id=f"{session_id}_w{i:05d}",
            session_id=session_id,
            t0=0.0,
            t1=duration,
            num_channels=n,
        )
        for i, (duration, n) in enumerate(windows)
    ]
    return SimpleNamespace(records=records, fs=fs, sessions=())


def combined(components):
    return SimpleNamespace(datasets=components)


def batch_ids(batches):
    """Sortable ``(component_idx, window_id)`` signature per batch.

    The spec-yield contract's per-window objects (``WindowRecord`` or the
    ``fake_dataset`` stand-ins) support ``==`` but not ``<``, so composition
    comparisons below sort on this projection instead of the raw specs.
    """
    return sorted(tuple((i, w.window_id) for i, w in batch) for batch in batches)


def window_ids(batches):
    return {w.window_id for batch in batches for _, w in batch}


# --------------------------------------------------------------------------
# record_costs / pack_batches: metadata-only cost model, packing budget.
# --------------------------------------------------------------------------


def test_record_costs_metadata_only():
    dataset = fake_dataset([(10.0, 2), (60.0, 3)])
    assert record_costs(dataset.records, FS) == [(240000, 2), (1440000, 3)]


def test_pack_batches_budget_and_coverage():
    windows = [(10.0 + i, 2) for i in range(10)] + [(60.0, 2), (5.0, 3)]
    costs = record_costs(fake_dataset(windows).records, FS)
    bins = 2 * 3 * round(FS * 20.0)  # about three 20 s N=2 windows
    batches = pack_batches(costs, bins, min_batch_size=1)

    seen = sorted(i for batch in batches for i in batch)
    assert seen == list(range(len(windows)))  # exactly once each
    for batch in batches:
        rows = sum(costs[i][1] for i in batch)
        t_max = max(costs[i][0] for i in batch)
        assert rows * t_max <= bins or len(batch) == 1


def test_pack_batches_min_batch_size():
    costs = record_costs(fake_dataset([(30.0, 2)] * 7).records, FS)
    batches = pack_batches(costs, batch_bins=1, min_batch_size=2)
    assert all(len(batch) >= 2 for batch in batches)
    assert sorted(i for b in batches for i in b) == list(range(7))


def test_no_weights_path_uses_each_components_own_fs():
    # b's windows are the same wall-clock duration as a's but at a third of
    # the sample rate; if the no-weights path used a single (wrong) fs for
    # both components instead of each component's own, the cost-sorted pack
    # order would diverge from this hand-computed expectation.
    a = fake_dataset([(2.0, 1), (2.0, 1)], session_id="a", fs=24000)
    b = fake_dataset([(2.0, 1), (2.0, 1)], session_id="b", fs=8000)
    bins = 2 * round(24000 * 2.0)

    sampler = ConversationBatchSampler(combined([a, b]), batch_bins=bins)
    got = [[(i, w.window_id) for i, w in batch] for batch in sampler]

    correct_costs = record_costs(a.records, 24000) + record_costs(b.records, 8000)
    specs = [(0, r) for r in a.records] + [(1, r) for r in b.records]
    packed = pack_batches(correct_costs, bins, min_batch_size=1)
    expected = [
        [(i, r.window_id) for i, r in (specs[j] for j in batch)] for batch in packed
    ]
    assert got == expected


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
    assert batch_ids(epoch0) == batch_ids(epoch1)  # same composition, different order


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
    ids0, ids1 = window_ids(rank0), window_ids(rank1)
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
    needs (see ``record_costs``), backing random tensors for a real
    LightningModule/Trainer fit - deliberately NOT ConversationDataset or the
    F5 model, which this test has no reason to drag in.  ``__getitem__``
    unpacks the sampler's ``(component_idx, record)`` spec, recovering the
    original row via each record's ``idx`` field."""

    def __init__(self, n: int = 8, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 4, generator=gen)
        self.y = torch.randn(n, 1, generator=gen)
        self.records = [
            SimpleNamespace(
                window_id=f"tiny_w{i:03d}",
                session_id="tiny",
                t0=0.0,
                t1=1.0 + i,
                num_channels=1,
                idx=i,
            )
            for i in range(n)
        ]
        self.fs = 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, spec):
        _, record = spec
        return self.x[record.idx], self.y[record.idx]


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


# --------------------------------------------------------------------------
# Sanity-probe worker-leak fix: ``num_sanity_val_steps`` is kept ON (2 of 6
# val batches) so a broken val/train path still fails fast at t=0, but the
# 2-batch sanity pass abandons an unexhausted CombinedLoader iterator whose
# (num_workers=2) worker processes would otherwise linger until the first
# real validation rebuilds it.  ``ConversationalLightningModule`` releases
# that iterator from ``on_validation_end`` (gated on
# ``trainer.sanity_checking``) - NOT ``on_sanity_check_end``, which is a
# Callback-only hook in Lightning 2.6.5 and never fires on a
# LightningModule (``hasattr(pl.LightningModule, "on_sanity_check_end")``
# is False; a live fit with that method defined never calls it).
# --------------------------------------------------------------------------


class _TinySupervisedDataset(torch.utils.data.Dataset):
    """Plain (non-conversation) tensor dataset for the val/train DataLoaders
    in the sanity-cleanup harness - deliberately simpler than
    ``_TinyConversationDataset``: this harness exercises the sanity/
    validation hook timing, not ``ConversationBatchSampler``."""

    def __init__(self, n: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 4, generator=gen)
        self.y = torch.randn(n, 1, generator=gen)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class _SanityCleanupModule(pl.LightningModule):
    """Minimal LightningModule exercising the sanity-cleanup fix without
    dragging in the full F5 model stack.

    ``on_validation_end`` delegates to
    ``ConversationalLightningModule._release_sanity_val_iterator`` (unbound,
    the same delegation pattern ``_ResumeOrderingModule`` uses for
    ``_initial_epoch``): that private helper only reads
    ``self.trainer``/``self._trainer`` state, so calling it on a module that
    is not actually a ``ConversationalLightningModule`` is safe, and this
    test fails if the production helper regresses instead of only checking
    a hand-rolled copy of it.  (The public ``on_validation_end`` hook itself
    is not delegated to directly: it calls ``super().on_validation_end()``,
    and a zero-arg ``super()`` compiled into ``ConversationalLightningModule``
    would break when invoked unbound on a module outside that MRO.)
    """

    def __init__(self, train_dataset, val_dataset):
        super().__init__()
        self.model = torch.nn.Linear(4, 1)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.sanity_validation_step_calls = 0
        self.real_validation_step_calls = 0

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, num_workers=0
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, num_workers=2
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            self.sanity_validation_step_calls += 1
        else:
            self.real_validation_step_calls += 1
        x, y = batch
        return torch.nn.functional.mse_loss(self.model(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def on_validation_end(self) -> None:
        if self.trainer.sanity_checking:
            ConversationalLightningModule._release_sanity_val_iterator(self)


class _RecordValCombinedLoaderIterator(pl.Callback):
    """Records the val loop's ``CombinedLoader._iterator`` at
    ``on_train_start`` - i.e. after the sanity pass has finished but before
    the first real training/validation step runs - so the test can tell
    whether the sanity pass's abandoned iterator was released in time."""

    def __init__(self):
        self.iterator_at_train_start = "not_recorded"

    def on_train_start(self, trainer, pl_module):
        combined_loader = trainer.fit_loop.epoch_loop.val_loop._combined_loader
        self.iterator_at_train_start = (
            None if combined_loader is None else combined_loader._iterator
        )


def test_sanity_check_end_releases_abandoned_val_iterator(tmp_path):
    """Regression test for the sanity-check worker leak.

    ``num_sanity_val_steps=2`` against a 6-batch, ``num_workers=2`` val
    DataLoader means the sanity pass consumes only 2 of 6 batches, leaving
    the ``CombinedLoader``'s iterator - and its worker processes - alive
    and unexhausted unless something explicitly releases it before the
    first real validation would otherwise rebuild it.
    ``ConversationalLightningModule.on_validation_end`` does that release
    via ``CombinedLoader.reset()`` (Lightning's own worker-shutdown path).

    ``CombinedLoader.reset()`` only clears the iterator, not the loader
    object itself, so the first real validation's ``setup_data()`` takes the
    "already built" early-return path and calls ``reset()`` -> ``iter()``
    again on the SAME ``CombinedLoader`` - a released iterator must still be
    usable, not just gone.  ``check_val_every_n_epoch`` defaults to 1, so
    this one-epoch fit also runs one real (non-sanity) validation pass,
    letting this test assert that rebuild succeeds in the same run.

    TDD evidence for this fix (see the report for the full transcript): with
    the ``on_validation_end`` override removed from ``_SanityCleanupModule``
    (the pre-fix state - no LightningModule-level cleanup at all, matching
    the codebase before this test was written), ``iterator_at_train_start``
    is NOT None: the sanity pass's abandoned iterator survives, unexhausted,
    into training. With the override delegating to the real
    ``ConversationalLightningModule._release_sanity_val_iterator`` (asserted
    below), it is None.
    """
    train_dataset = _TinySupervisedDataset(n=4, seed=0)
    val_dataset = _TinySupervisedDataset(n=6, seed=1)
    module = _SanityCleanupModule(train_dataset, val_dataset)
    recorder = _RecordValCombinedLoaderIterator()
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=2,
        limit_train_batches=1,
        limit_val_batches=6,
        use_distributed_sampler=False,
        callbacks=[recorder],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(module)

    assert module.sanity_validation_step_calls >= 1  # the sanity probe really ran
    assert recorder.iterator_at_train_start is None  # its iterator was released
    # ... and the released loader still works: the first real validation
    # (check_val_every_n_epoch=1) rebuilt and iterated it successfully.
    assert module.real_validation_step_calls == 6


# --------------------------------------------------------------------------
# Per-corpus weighted batch interleaving (Task 3): weights=None must stay
# byte-identical (in composition/order) to the pre-existing global-packing
# path; weights=[...] packs each CombinedDataset component separately and
# interleaves per-epoch quotas drawn by seeded permutation.
# --------------------------------------------------------------------------


def batch_signature(batches):
    """(duration, num_channels) per window per batch - a component-idx- and
    identity-agnostic projection used to compare packing composition/order
    across sampler inputs shaped differently (single dataset vs multiple
    CombinedDataset components) but expected to pack identically."""
    return [
        [(round(w.t1 - w.t0, 6), w.num_channels) for _, w in batch] for batch in batches
    ]


def test_weights_none_multi_component_matches_global_packing():
    a = fake_dataset([(10.0, 2)] * 4, session_id="a")
    b = fake_dataset([(5.0, 1)] * 6, session_id="b")
    merged = fake_dataset([(10.0, 2)] * 4 + [(5.0, 1)] * 6, session_id="m")
    bins = 2 * round(FS * 30.0)
    with_none = ConversationBatchSampler(combined([a, b]), batch_bins=bins)
    flat = ConversationBatchSampler(merged, batch_bins=bins)
    assert batch_signature(list(with_none)) == batch_signature(list(flat))


def test_weighted_quotas_and_composition():
    # one 10 s N=2 window per batch at these bins
    bins = 2 * round(FS * 10.0)
    a = fake_dataset([(10.0, 2)] * 6, session_id="a")  # 6 batches
    b = fake_dataset([(10.0, 2)] * 24, session_id="b")  # 24 batches
    sampler = ConversationBatchSampler(
        combined([a, b]), batch_bins=bins, weights=[0.5, 0.5], shuffle=True, seed=0
    )
    batches = list(sampler)
    # L = min(6/0.5, 24/0.5) = 12 -> quotas [6, 6]
    assert len(batches) == 12
    from_a = [batch for batch in batches if {i for i, _ in batch} == {0}]
    from_b = [batch for batch in batches if {i for i, _ in batch} == {1}]
    assert len(from_a) == 6  # ALL of the scarcer-relative-to-weight corpus
    assert len(from_b) == 6
    for batch in batches:  # corpus-pure batches, structurally guaranteed
        assert len({i for i, _ in batch}) == 1


def test_weighted_rotation_and_constant_length():
    bins = 2 * round(FS * 10.0)
    a = fake_dataset([(10.0, 2)] * 6, session_id="a")
    b = fake_dataset([(10.0, 2)] * 24, session_id="b")
    sampler = ConversationBatchSampler(
        combined([a, b]), batch_bins=bins, weights=[0.5, 0.5], shuffle=True, seed=0
    )

    def epoch_b_window_ids(epoch):
        sampler.set_epoch(epoch)
        return sorted(
            tuple(sorted(w.window_id for _, w in batch))
            for batch in sampler
            if {i for i, _ in batch} == {1}
        )

    sampler.set_epoch(0)
    len0 = len(sampler)
    sampler.set_epoch(1)
    assert len(sampler) == len0  # constant epoch length
    assert epoch_b_window_ids(0) != epoch_b_window_ids(1)  # B subset rotates
    assert epoch_b_window_ids(0) == epoch_b_window_ids(0)  # deterministic per epoch


def test_weight_zero_is_sole_corpus_training():
    bins = 2 * round(FS * 10.0)
    a = fake_dataset([(10.0, 2)] * 5, session_id="a")
    b = fake_dataset([(12.0, 2)] * 7, session_id="b")
    only_a = ConversationBatchSampler(
        combined([a, b]), batch_bins=bins, weights=[1.0, 0.0]
    )
    solo = ConversationBatchSampler(a, batch_bins=bins)
    assert batch_ids(list(only_a)) == batch_ids(list(solo))


def test_three_component_weights():
    bins = 2 * round(FS * 10.0)
    comps = [
        fake_dataset([(10.0, 2)] * n, session_id=f"c{k}")
        for k, n in enumerate((4, 8, 16))
    ]
    sampler = ConversationBatchSampler(
        combined(comps), batch_bins=bins, weights=[0.25, 0.25, 0.5]
    )
    # L = min(4/.25, 8/.25, 16/.5) = 16 -> quotas [4, 4, 8]
    assert len(list(sampler)) == 16


def test_four_component_weights():
    bins = 2 * round(FS * 10.0)
    comps = [
        fake_dataset([(10.0, 2)] * n, session_id=f"c{k}")
        for k, n in enumerate((4, 8, 16, 8))
    ]
    sampler = ConversationBatchSampler(
        combined(comps), batch_bins=bins, weights=[0.25, 0.25, 0.25, 0.25]
    )
    # L = min(4/.25, 8/.25, 16/.25, 8/.25) = 16 -> quotas [4, 4, 4, 4]
    assert len(list(sampler)) == 16


def test_weights_validation():
    a = fake_dataset([(10.0, 2)] * 2, session_id="a")
    b = fake_dataset([(10.0, 2)] * 2, session_id="b")
    bins = 2 * round(FS * 10.0)
    with pytest.raises(ValueError):  # wrong length
        ConversationBatchSampler(combined([a, b]), batch_bins=bins, weights=[1.0])
    with pytest.raises(ValueError):  # negative
        ConversationBatchSampler(combined([a, b]), batch_bins=bins, weights=[1.0, -0.1])
    with pytest.raises(ValueError):  # all zero
        ConversationBatchSampler(combined([a, b]), batch_bins=bins, weights=[0.0, 0.0])


def test_weighted_empty_component_with_positive_weight_raises():
    # An empty/truncated manifest packs to zero batches; with p_i > 0 the old
    # epoch_len = min(n_i / p_i) collapsed to 0 and quotas floored to
    # max(1, ...) = 1, silently shrinking the epoch to ~1 batch. This must
    # raise instead, naming the offending component index. Packing now
    # happens lazily (on first plan, not at construction), so the raise
    # surfaces on iteration/len, not ConversationBatchSampler(...) itself.
    a = fake_dataset([(10.0, 2)] * 6, session_id="a")
    b = fake_dataset([], session_id="b")
    bins = 2 * round(FS * 10.0)
    sampler = ConversationBatchSampler(
        combined([a, b]), batch_bins=bins, weights=[0.5, 0.5]
    )
    with pytest.raises(ValueError, match=r"component 1"):
        list(sampler)


def test_weighted_empty_component_with_zero_weight_is_excluded():
    # Weight 0.0 on the empty component must still work: it is excluded from
    # packing/quota entirely, so an empty manifest for a disabled corpus is
    # not an error.
    a = fake_dataset([(10.0, 2)] * 6, session_id="a")
    b = fake_dataset([], session_id="b")
    bins = 2 * round(FS * 10.0)
    sampler = ConversationBatchSampler(
        combined([a, b]), batch_bins=bins, weights=[1.0, 0.0]
    )
    solo = ConversationBatchSampler(a, batch_bins=bins)
    assert batch_ids(list(sampler)) == batch_ids(list(solo))


# --------------------------------------------------------------------------
# Task 9: online per-epoch planning. mixed_dataset/synthetic_200_sessions
# below back these with real ConversationDataset/SessionRecord objects (no
# audio files needed - the sampler never calls __getitem__/loads audio, only
# plan_windows/.records/.fs/.sessions).
# --------------------------------------------------------------------------


def _alternating_turns(session_id, num_channels, duration, utt_len=2.5, gap=1.5):
    """Round-robin turns across channels with clean inter-turn silences,
    enough coverage for the real windowing planner to produce several
    windows per session (mirrors dataset/tests/conftest.py's
    ``_alternating_sups``, but yields ``Turn`` objects directly)."""
    turns = []
    t, i = 0.5, 0
    texts = ["can you hear me", "yes loud and clear", "great let us start"]
    while t + utt_len + 0.5 < duration:
        channel = i % num_channels
        turns.append(
            Turn(
                channel=channel,
                speaker=f"{session_id}_spk{channel}",
                text=texts[i % len(texts)],
                start=round(t, 3),
                end=round(t + utt_len, 3),
            )
        )
        t += utt_len + gap
        i += 1
    return turns


def _session_records(prefix, num_sessions, num_channels, duration, **turn_kwargs):
    return [
        SessionRecord(
            session_id=f"{prefix}_sess{k}",
            audio_relpath=f"original/{prefix}_sess{k}_mixed.flac",
            num_channels=num_channels,
            sample_rate=48000,
            duration=duration,
            turns=tuple(
                _alternating_turns(
                    f"{prefix}_sess{k}", num_channels, duration, **turn_kwargs
                )
            ),
        )
        for k in range(num_sessions)
    ]


def _conversation_dataset(tmp_path, prefix, sessions, **kw):
    manifest = tmp_path / f"{prefix}_sessions.jsonl"
    write_session_manifest(manifest, sessions)
    defaults = dict(
        split="train",
        manifest_path=manifest,
        dataset_root=tmp_path,
        fs=FS,
        window_params={"window_min": 4.0, "window_max": 10.0, "tail_min": 2.0},
    )
    defaults.update(kw)
    return ConversationDataset(**defaults)


@pytest.fixture
def mixed_dataset(tmp_path):
    """Two independent ConversationDataset components. Sessions are several
    multiples of window_max (60s/40s vs a 10s window_max) so per-epoch RNG
    actually moves cut points (a session <= window_max always emits exactly
    one boundary-independent tail span, so short sessions could not
    distinguish epochs)."""
    comp0 = _conversation_dataset(
        tmp_path / "c0",
        "c0",
        _session_records("c0", num_sessions=3, num_channels=2, duration=60.0),
    )
    comp1 = _conversation_dataset(
        tmp_path / "c1",
        "c1",
        _session_records("c1", num_sessions=3, num_channels=3, duration=40.0),
    )
    return combined([comp0, comp1])


@pytest.fixture
def synthetic_200_sessions():
    """200 in-memory SessionRecords, ~100 turns each, no manifest/audio I/O -
    a regression canary for per-epoch planning cost, not a benchmark."""
    sessions = []
    for k in range(200):
        session_id = f"speed_sess{k}"
        turns = tuple(
            _alternating_turns(
                session_id, num_channels=2, duration=300.0, utt_len=1.6, gap=1.2
            )
        )
        sessions.append(
            SessionRecord(
                session_id=session_id,
                audio_relpath=f"original/{session_id}_mixed.flac",
                num_channels=2,
                sample_rate=48000,
                duration=300.0,
                turns=turns,
            )
        )
    return sessions


class _StubDataset:
    """Duck-typed ConversationBatchSampler input: plan_windows/records/fs/
    sessions only, no on-disk manifest or audio. Used ONLY by the planning-
    speed regression test below; every other test needing a real dataset
    uses ConversationDataset via the mixed_dataset fixture."""

    def __init__(self, sessions, window_params, seed=0):
        self.sessions = sessions
        self.fs = FS
        self._window_params = window_params
        self._seed = seed
        self.records = self.plan_windows(None)

    def plan_windows(self, epoch):
        records, _stats = plan_sessions(
            self.sessions, params=self._window_params, seed=self._seed, epoch=epoch
        )
        return records


def _dataset_over(sessions):
    return _StubDataset(sessions, WindowParams())


class TestOnlineMode:
    def test_epoch_changes_batches(self, mixed_dataset):
        s = ConversationBatchSampler(
            mixed_dataset, batch_bins=BB, online=True, shuffle=True
        )

        def spans(epoch):
            s.set_epoch(epoch)
            return {(w.session_id, w.t0, w.t1) for b in s for _, w in b}

        # spans(0) a second time in a row would just hit the (0, True) cache
        # and return by identity - never able to fail. Evict it first by
        # visiting epoch 1, so the re-derived epoch-0 plan is a fresh
        # _plan_and_pack call, pinning the resume contract for real.
        s0 = spans(0)
        s1 = spans(1)
        assert s0 == spans(0)  # re-derivable after cache eviction (resume contract)
        assert s0 != s1  # fresh windows across epochs

    def test_frozen_mode_matches_records(self, mixed_dataset):
        s = ConversationBatchSampler(mixed_dataset, batch_bins=BB, online=False)
        yielded = sorted(w.window_id for b in s for _, w in b)
        expected = sorted(
            r.window_id for c in mixed_dataset.datasets for r in c.records
        )
        assert yielded == expected  # valid split == frozen plan, all of it

    def test_batch_bins_respected(self, mixed_dataset):
        s = ConversationBatchSampler(mixed_dataset, batch_bins=BB, online=True)
        for batch in s:
            costs = [(round(FS * (w.t1 - w.t0)), w.num_channels) for _, w in batch]
            rows, tmax = sum(n for _, n in costs), max(t for t, _ in costs)
            assert rows * tmax <= BB or len(batch) == 1  # min_batch_size escape

    def test_weights_quotas_still_apply(self, mixed_dataset):
        # weight 0 excludes a corpus entirely; nonzero weights yield batches
        # from that component only, never mixed within a batch
        s = ConversationBatchSampler(
            mixed_dataset, batch_bins=BB, online=True, weights=[1.0, 0.0]
        )
        s.set_epoch(0)
        comp_ids = {i for b in s for i, _ in b}
        assert comp_ids == {0}
        for batch in s:
            assert len({i for i, _ in batch}) == 1  # no cross-corpus batches

    def test_weighted_quota_recomputed_each_online_epoch(self, mixed_dataset):
        # weights=[1.0, 0.0] above never exercises the permutation branch in
        # _epoch_batches (the nonzero component's quota always covers its
        # whole batch count when the other component is excluded), and the
        # frozen-mode weighted tests use per-component counts that never
        # move. This test forces both: two nonzero-weight components whose
        # online-mode plan (and hence packed batch count) changes epoch to
        # epoch, so a stale-quota bug - the exact failure mode per-epoch
        # planning introduces - would be caught.
        bins = BB
        probs = [0.5, 0.5]
        s = ConversationBatchSampler(
            mixed_dataset,
            batch_bins=bins,
            online=True,
            weights=probs,
            shuffle=True,
            seed=0,
        )

        ran_permutation = False
        for epoch in (0, 1):
            # Recompute the expected quota straight from THAT epoch's plan,
            # independently of the sampler under test (same formula the
            # sampler itself uses, applied to a fresh plan_windows/pack_batches
            # call here).
            comp_batch_counts = [
                len(
                    pack_batches(
                        record_costs(c.plan_windows(epoch), int(c.fs)),
                        bins,
                        min_batch_size=1,
                    )
                )
                for c in mixed_dataset.datasets
            ]
            epoch_len = min(n / p for n, p in zip(comp_batch_counts, probs) if p > 0)
            expected_quotas = [max(1, round(p * epoch_len)) for p in probs]
            if any(q < n for q, n in zip(expected_quotas, comp_batch_counts)):
                ran_permutation = True

            s.set_epoch(epoch)
            batches = list(s)
            actual_quotas = [0] * len(probs)
            for batch in batches:
                comp_ids = {i for i, _ in batch}
                assert len(comp_ids) == 1  # never mixed
                actual_quotas[comp_ids.pop()] += 1
            assert actual_quotas == expected_quotas

            # len/iter agreement in the same epoch: not obvious once the
            # permutation draw is in the mix (a stale-quota bug could easily
            # make __len__ and __iter__ disagree instead of just being wrong
            # together).
            assert len(s) == len(list(s))

        assert ran_permutation, (
            "fixture/bins must exercise the permutation branch (a quota "
            "strictly below its component's batch count) in at least one "
            "epoch, or this test cannot distinguish a stale-quota bug"
        )

    def test_component_idx_is_correct(self, mixed_dataset):
        s = ConversationBatchSampler(mixed_dataset, batch_bins=BB, online=True)
        s.set_epoch(2)
        session_ids_per_comp = [
            {sess.session_id for sess in c.sessions} for c in mixed_dataset.datasets
        ]
        for batch in s:
            for comp_idx, record in batch:
                assert record.session_id in session_ids_per_comp[comp_idx]

    def test_plan_logs_timing_info(self, mixed_dataset, caplog):
        s = ConversationBatchSampler(mixed_dataset, batch_bins=BB, online=True)
        with caplog.at_level(
            logging.INFO, logger="egs3.conversational.tts.src.sampler"
        ):
            s.set_epoch(0)
            len(s)
        lines = [r.message for r in caplog.records if "window plan" in r.message]
        assert lines, "expected an INFO 'window plan' log line per fresh plan"
        line = lines[-1]
        assert "sessions" in line and "windows" in line
        assert "batches" in line
        assert "plan+pack" in line and "s" in line  # wall-seconds suffix


class TestPlanningSpeed:
    def test_epoch_plan_under_threshold(self, synthetic_200_sessions):
        # 200 in-memory sessions, ~100 turns each: plan + pack must stay
        # well under real-training irrelevance. Regression canary, not a
        # benchmark - threshold is deliberately loose.
        ds = _dataset_over(synthetic_200_sessions)
        s = ConversationBatchSampler(ds, batch_bins=BB, online=True)
        t0 = time.perf_counter()
        s.set_epoch(5)
        _ = len(s)
        assert time.perf_counter() - t0 < 2.0
