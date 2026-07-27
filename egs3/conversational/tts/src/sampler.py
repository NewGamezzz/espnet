"""Duration-bucketed batch sampler over conversation-window metadata.

The espnet3 iter_factory path was rejected for this recipe (see README):
its stock numel sampler needs shape files that must stay in sync with the
``min_active_speakers`` dataset filter, and
``DataLoaderBuilder._build_iter_factory`` calls ``build_iter(epoch,
shuffle=False)``, freezing the batch order across epochs regardless of the
config's ``shuffle: true``.  This sampler instead computes the same
inventory directly from the window manifests already held by
``ConversationDataset`` and plugs into a standard PyTorch ``DataLoader``
(built per epoch by ``src/lit_module.py``, which passes the current epoch
for seeded reshuffling).

Cost model (espnet3 numel strategy with a row-based cost): a window of N
channels and duration ``t1 - t0`` costs ``N`` transformer rows of
``round(fs * (t1 - t0))`` samples each; a batch padded to its longest
window costs ``total_rows * T_max`` sample-rows, and ``batch_bins`` is
sized directly in those units (~94 mel frames per second per row at 24 kHz
/ hop 256).  Shapes come from window metadata alone; no audio is loaded.
``min_batch_size`` counts conversations, not rows, and mixed-N batches need
no special casing because the budget is in N x T units.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import torch


def _component_datasets(dataset) -> Sequence:
    """The ``ConversationDataset``s behind ``dataset``, in index order.

    Accepts a bare ``ConversationDataset`` or espnet3's ``CombinedDataset``
    (whose global index concatenates the components).
    """
    return getattr(dataset, "datasets", None) or [dataset]


def window_costs(dataset) -> list[tuple[int, int]]:
    """Per global index: ``(T_samples, num_channels)`` from metadata alone."""
    costs: list[tuple[int, int]] = []
    for component in _component_datasets(dataset):
        records = getattr(component, "records", None)
        if records is None:
            raise TypeError(
                f"{type(component).__name__} has no window records; "
                "ConversationBatchSampler only works with ConversationDataset"
            )
        fs = int(component.fs)
        for record in records:
            costs.append((round(fs * (record.t1 - record.t0)), record.num_channels))
    return costs


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
    ``epoch`` is only the initial value.  ``__iter__`` yields lists of global
    dataset indices.  With ``shuffle``, the batch ORDER is shuffled with
    ``RandomState(seed + epoch)`` (batch composition stays fixed - duration
    bucketing is the point).  Under torch.distributed, tail batches are
    dropped so every rank sees the same batch count (the same policy as
    espnet3's iter_factory path), then the batches are strided by rank.
    """

    def __init__(
        self,
        dataset,
        batch_bins: int,
        min_batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self._packed = pack_batches(
            window_costs(dataset), batch_bins, min_batch_size=min_batch_size
        )
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

    def _epoch_batches(self) -> list[list[int]]:
        batches = list(self._packed)
        if self.shuffle:
            np.random.RandomState(self.seed + self.epoch).shuffle(batches)
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

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._epoch_batches())
