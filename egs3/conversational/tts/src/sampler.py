"""Duration-bucketed batch sampler over conversation-window metadata.

The espnet3 iter_factory path was rejected for this recipe (see README):
its stock numel sampler needs shape files that must stay in sync with the
``min_active_speakers`` dataset filter, and
``DataLoaderBuilder._build_iter_factory`` calls ``build_iter(epoch,
shuffle=False)``, freezing the batch order across epochs regardless of the
config's ``shuffle: true``.  This sampler instead computes the same
inventory directly from the window metadata already held by
``ConversationDataset`` and plugs into a standard PyTorch ``DataLoader``
(built per epoch by ``src/lit_module.py``, which passes the current epoch
for seeded reshuffling).

Windows are planned per epoch (``ConversationDataset.plan_windows``) rather
than pre-baked into a static manifest: with ``online=True`` the sampler calls
``plan_windows(epoch)`` fresh on every ``(epoch, online)`` cache miss, so
training sees new cut points every epoch - and hence a batch COUNT that can
vary slightly epoch to epoch - identically on every DDP rank and DataLoader
worker, because the plan is a pure function of ``(window_seed, epoch, session
metadata)``.  Lightning tolerates this because it re-queries ``__len__`` (via
``len(sampler)``/the progress bar total) at the start of every epoch rather
than caching it once for the whole fit.  ``online=False`` (the default, used
for valid/test splits and inference) instead reads each component's frozen
``.records`` inventory - the exact windows and order emitted before per-epoch
planning existed - so the batch count there is constant across epochs, as
before.

``__iter__`` always yields lists of ``(component_idx, WindowRecord)`` specs -
one uniform contract across both frozen and online modes - instead of plain
dataset indices: online windows are planned fresh each epoch and have no
fixed global index for a ``Dataset.__getitem__(int)`` to look up, so the
downstream ``Dataset`` must accept the spec directly.

Cost model (espnet3 numel strategy with a row-based cost): a window of N
channels and duration ``t1 - t0`` costs ``N`` transformer rows of
``round(fs * (t1 - t0))`` samples each; a batch padded to its longest
window costs ``total_rows * T_max`` sample-rows, and ``batch_bins`` is
sized directly in those units (~94 mel frames per second per row at 24 kHz
/ hop 256).  Shapes come from window metadata alone; no audio is loaded.
``min_batch_size`` counts conversations, not rows, and mixed-N batches need
no special casing because the budget is in N x T units.  With the
``weights`` knob (see ``ConversationBatchSampler``), each corpus component is
packed under this same cost model independently, so batches never mix
windows from different corpora; the no-weights path computes costs per
component too, at THAT component's own training ``fs``, before concatenating
the cost lists ahead of one global pack, since components can run at
different rates.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from egs3.conversational.tts.dataset.preprocessing.windows import WindowRecord

logger = logging.getLogger(__name__)


def _component_datasets(dataset) -> Sequence:
    """The ``ConversationDataset``s behind ``dataset``, in index order.

    Accepts a bare ``ConversationDataset`` or espnet3's ``CombinedDataset``
    (whose global index concatenates the components).
    """
    return getattr(dataset, "datasets", None) or [dataset]


def record_costs(records: Sequence["WindowRecord"], fs: int) -> list[tuple[int, int]]:
    """Per record: ``(T_samples, num_channels)`` from metadata alone."""
    return [(round(fs * (r.t1 - r.t0)), r.num_channels) for r in records]


def pack_batches(
    costs: Sequence[tuple[int, int]],
    batch_bins: int,
    min_batch_size: int = 1,
) -> list[list[int]]:
    """Greedy duration-sorted packing under the padded row-sample budget.

    Windows are sorted by duration (ties by index) and accumulated while the
    padded cost ``(rows_so_far + N_i) * T_i`` stays within ``batch_bins``
    (ascending order makes ``T_i`` the batch's padded length).  A batch only
    closes once it holds ``min_batch_size`` conversations, so single batches
    may exceed the budget when the two constraints conflict; a short final
    batch is redistributed over the preceding ones like espnet2's sampler.
    """
    if batch_bins <= 0:
        raise ValueError(f"batch_bins must be positive, got {batch_bins}")
    order = sorted(range(len(costs)), key=lambda i: (costs[i][0], i))
    batches: list[list[int]] = []
    batch: list[int] = []
    rows = 0
    for i in order:
        t_i, n_i = costs[i]
        if batch and (rows + n_i) * t_i > batch_bins and len(batch) >= min_batch_size:
            batches.append(batch)
            batch, rows = [], 0
        batch.append(i)
        rows += n_i
    if batch:
        if len(batch) < min_batch_size and batches:
            for k, idx in enumerate(batch):
                batches[-(k % len(batches)) - 1].append(idx)
        else:
            batches.append(batch)
    return batches


class ConversationBatchSampler:
    """Batch sampler with seeded per-epoch reshuffling and DDP alignment.

    Reshuffling is driven by ``set_epoch`` between epochs; the constructor
    ``epoch`` is only the initial value.  ``__iter__`` yields lists of
    ``(component_idx, WindowRecord)`` specs.  With ``shuffle``, the batch
    ORDER is shuffled with ``RandomState(seed + epoch)`` (batch composition
    within online-mode plans stays fixed once planned for that epoch -
    duration bucketing is the point).  Under torch.distributed, tail batches
    are dropped so every rank sees the same batch count (the same policy as
    espnet3's iter_factory path), then the batches are strided by rank.

    ``online`` selects the planning source: ``False`` (default) reads each
    component's frozen ``.records`` inventory, unchanged across epochs -
    valid/test splits and inference want the same windows every time.
    ``True`` calls ``component.plan_windows(epoch)`` fresh on every epoch
    change, so training sees new cut points; the plan (and its resulting
    pack) is cached on ``(self.epoch, self.online)`` and only recomputed on a
    cache miss, with the planning + packing wall time and window/batch counts
    logged at INFO so per-epoch planning cost stays visible.

    ``weights`` is an optional list of non-negative floats, one per
    ``CombinedDataset`` component in config order (the ``dataloader.train.weights``
    config knob), giving each corpus a target fraction of optimizer steps.
    ``None`` (the default) keeps the legacy global-packing path: one
    duration-bucketed pack over the concatenated inventory, byte-identical to
    the sampler before this knob existed.  With weights, each component is
    packed separately and normalized into probabilities ``p_i``; the epoch
    length is ``L = min over p_i > 0 of (n_i / p_i)`` batches (the corpus that
    is scarcest relative to its weight is fully covered every epoch, and no
    corpus is ever asked for more batches than it has), each component's
    per-epoch quota is ``max(1, round(p_i * L))`` batches (``0`` if
    ``p_i == 0``, i.e. the corpus is excluded that epoch), and the quota
    subset is drawn by seeded permutation (``RandomState(seed + epoch)``,
    rotating which batches are picked across epochs).  In ``online`` mode the
    per-component batch counts ``n_i`` are fresh every epoch (new plans pack
    to slightly different counts), so quotas are recomputed every epoch from
    the current plan rather than fixed once at construction; the quota
    FORMULA itself never changes.
    """

    def __init__(
        self,
        dataset,
        batch_bins: int,
        min_batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
        weights: Sequence[float] | None = None,
        online: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_bins = batch_bins
        self.min_batch_size = min_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.online = online
        self.weights = None if weights is None else list(weights)
        if self.weights is not None:
            components = _component_datasets(dataset)
            if len(self.weights) != len(components):
                raise ValueError(
                    f"{len(self.weights)} weights for {len(components)} dataset "
                    "components; weights align with the config's train entries"
                )
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError(
                    f"weights must be non-negative with a positive sum, got "
                    f"{self.weights}"
                )
            total = sum(self.weights)
            self._probs = [w / total for w in self.weights]
        else:
            self._probs = None
        # Per-epoch plan+pack cache: (epoch, online) -> (batches, component_batches).
        # set_epoch needs no extra invalidation logic beyond changing self.epoch,
        # which changes the cache key.
        self._cache_key: tuple[int | None, bool | None] = (None, None)
        self._cache_batches = (None, None)
        # Lightning's per-epoch epoch propagation (_set_sampler_epoch) only
        # looks at dataloader.sampler and dataloader.batch_sampler.sampler,
        # never at the batch sampler itself, so expose self under .sampler.
        self.sampler = self

    def set_epoch(self, epoch: int) -> None:
        """Adopt ``epoch`` for the next ``__iter__`` (DistributedSampler contract).

        Lightning calls this at every epoch start with the number of completed
        epochs, which equals the ``current_epoch`` the old per-epoch-rebuild
        scheme passed to the constructor, so batch order is unchanged.
        """
        self.epoch = int(epoch)

    @staticmethod
    def _world_info() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _plan_and_pack(self):
        """Plan this epoch's windows and pack them into batches, cached on
        ``(self.epoch, self.online)``.

        Returns ``(batches, component_batches)``: exactly one of the two is
        ``None`` depending on ``self.weights`` (mirrors the two packing paths
        below - the no-weights path fills ``batches``, the weights path fills
        ``component_batches``).
        """
        key = (self.epoch, self.online)
        if self._cache_key == key:
            return self._cache_batches
        t_start = time.perf_counter()
        components = _component_datasets(self.dataset)
        epoch_arg = self.epoch if self.online else None
        plans = [
            c.plan_windows(epoch_arg) if self.online else list(c.records)
            for c in components
        ]
        if self.weights is None:
            # Global packing over the concatenated inventory: byte-identical
            # composition to the sampler before the weights knob existed.
            # Each component can run at a different training fs, so costs
            # are computed per component (at that component's own fs) before
            # the cost lists are merged ahead of one global pack.
            specs: list[tuple[int, "WindowRecord"]] = []
            costs: list[tuple[int, int]] = []
            for i, (component, plan) in enumerate(zip(components, plans)):
                specs.extend((i, r) for r in plan)
                costs.extend(record_costs(plan, int(component.fs)))
            packed = pack_batches(costs, self.batch_bins, self.min_batch_size)
            batches = [[specs[j] for j in batch] for batch in packed]
            component_batches = None
        else:
            component_batches = []
            for i, (component, plan) in enumerate(zip(components, plans)):
                packed = pack_batches(
                    record_costs(plan, int(component.fs)),
                    self.batch_bins,
                    self.min_batch_size,
                )
                if self._probs[i] > 0 and not packed:
                    raise ValueError(
                        f"component {i} has weight {self.weights[i]} > 0 but "
                        "packed to zero batches (empty or truncated manifest?); "
                        "fix the manifest or set its weight to 0.0 to exclude it"
                    )
                component_batches.append([[(i, plan[j]) for j in b] for b in packed])
            batches = None
        elapsed = time.perf_counter() - t_start
        n_batches = (
            len(batches)
            if batches is not None
            else sum(len(b) for b in component_batches)
        )
        logger.info(
            "window plan epoch=%s online=%s: %d sessions -> %d windows, "
            "%d batches, plan+pack %.2fs",
            self.epoch,
            self.online,
            sum(len(getattr(c, "sessions", ())) for c in components),
            sum(len(p) for p in plans),
            n_batches,
            elapsed,
        )
        self._cache_key = key
        self._cache_batches = (batches, component_batches)
        return self._cache_batches

    def _epoch_batches(self) -> list[list[tuple[int, "WindowRecord"]]]:
        batches, component_batches = self._plan_and_pack()
        if component_batches is None:
            batches = list(batches)
            if self.shuffle:
                np.random.RandomState(self.seed + self.epoch).shuffle(batches)
        else:
            # One RandomState drives both the per-corpus quota draw and the
            # interleave shuffle, so every rank computes the same epoch.
            # Quotas are recomputed from THIS epoch's per-component batch
            # counts (constant in frozen mode, fresh every epoch in online
            # mode); the formula itself is unchanged either way.
            rng = np.random.RandomState(self.seed + self.epoch)
            epoch_len = min(
                len(comp_batches) / p
                for comp_batches, p in zip(component_batches, self._probs)
                if p > 0
            )
            quotas = [max(1, round(p * epoch_len)) if p > 0 else 0 for p in self._probs]
            batches = []
            for comp_batches, quota in zip(component_batches, quotas):
                if quota >= len(comp_batches):
                    batches.extend(comp_batches)
                else:
                    picks = rng.permutation(len(comp_batches))[:quota]
                    batches.extend(comp_batches[i] for i in picks)
            if self.shuffle:
                rng.shuffle(batches)
        rank, world_size = self._world_info()
        if world_size > 1:
            if len(batches) < world_size:
                raise RuntimeError(
                    f"{len(batches)} batches < world_size {world_size}; "
                    "increase the dataset or reduce batch_bins"
                )
            keep = len(batches) - len(batches) % world_size
            batches = batches[:keep][rank::world_size]
        return batches

    def __len__(self) -> int:
        return len(self._epoch_batches())

    def __iter__(self) -> Iterator[list[tuple[int, "WindowRecord"]]]:
        return iter(self._epoch_batches())
