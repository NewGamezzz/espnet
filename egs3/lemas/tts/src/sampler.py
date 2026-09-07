"""Batches that never leave one pack block, so a batch costs one 4 MB read.

The packs are chunk-contiguous (``dataset/extract.py``): a row's speaker-prompt
partners are its neighbours in pack order and its language prompt comes from
the neighbouring blocks (``LEMASDataset._draw_lang``). This sampler completes
the picture on the batching side: rows are grouped by the 4 MB block holding
their start, length-sorted inside each block and cut into numel batches
(``sum(frames x n_mels) <= batch_bins``, at least ``min_batch_size`` rows).
Batch order is shuffled per epoch and sharded by rank here, because the plain
torch DataLoader path is used (``iter_factory: null`` and
``trainer.use_distributed_sampler: false``).
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
import torch


def _unwrap(dataset):
    """Return the ``LEMASDataset`` inside espnet3's single-dataset wrappers.

    The wrappers (``CombinedDataset`` over one dataset) keep the index space of
    the inner dataset, so its pack columns can drive the batching directly.
    """
    seen = 0
    while not hasattr(dataset, "cols") and seen < 8:
        inner = getattr(dataset, "datasets", None) or getattr(dataset, "dataset", None)
        if isinstance(inner, (list, tuple)):
            if len(inner) != 1:
                raise ValueError("BlockBatchSampler needs a single underlying dataset")
            inner = inner[0]
        if inner is None:
            break
        dataset = inner
        seen += 1
    if not hasattr(dataset, "cols"):
        raise TypeError("BlockBatchSampler needs a LEMASDataset")
    return dataset


class BlockBatchSampler(torch.utils.data.Sampler):
    """Numel batches confined to pack blocks; see the module docstring."""

    def __init__(
        self,
        dataset,
        batch_bins: int,
        n_mels: int = 100,
        min_batch_size: int = 1,
        seed: int = 0,
        respect_blocks: Optional[bool] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """Build the per-row keys once; batches are cut lazily per epoch.

        Args:
            dataset: A ``LEMASDataset`` (needs ``cols``, ``cfg``, ``epoch``,
                ``train`` and ``frames_bound()``).
            batch_bins: Maximum ``sum(frames x n_mels)`` per batch.
            n_mels: Mel channels, the second factor of the numel cost.
            min_batch_size: Rows per batch at least (a block with fewer rows
                still yields its rows as one batch).
            seed: Shuffle seed, combined with the dataset epoch.
            respect_blocks: Confine batches to blocks. Defaults to
                ``dataset.train``: validation batches by length only, since a
                few hundred scattered validation rows are read once per check.
            rank: Rank for sharding; defaults to ``torch.distributed`` when
                initialised, else 0.
            world_size: World size for sharding; same default.
        """
        dataset = self.dataset = _unwrap(dataset)
        self.batch_bins = int(batch_bins)
        self.n_mels = int(n_mels)
        self.min_batch_size = max(1, int(min_batch_size))
        self.seed = int(seed)
        self.respect = bool(dataset.train) if respect_blocks is None else respect_blocks
        if rank is None or world_size is None:
            dist = torch.distributed
            if dist.is_available() and dist.is_initialized():
                rank, world_size = dist.get_rank(), dist.get_world_size()
            else:
                rank, world_size = 0, 1
        self.rank, self.world_size = int(rank), int(world_size)
        cols = dataset.cols
        block = int(dataset.cfg["block_samples"])
        self.cost = dataset.frames_bound() * self.n_mels
        self.block_key = cols.pack.astype(np.int64) * (1 << 40) + cols.a_start // block
        self._epoch: Optional[int] = None
        self._batches: List[np.ndarray] = []

    # ---- batching ------------------------------------------------------------
    def _cut(self) -> List[np.ndarray]:
        key = self.block_key if self.respect else np.zeros_like(self.block_key)
        order = np.lexsort((self.cost, key))
        k, c = key[order], self.cost[order]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        ends = np.r_[starts[1:], len(k)]
        batches: List[np.ndarray] = []
        bins, m = self.batch_bins, self.min_batch_size
        for s, e in zip(starts.tolist(), ends.tolist()):
            cs = np.cumsum(c[s:e])
            i = 0
            n = e - s
            while i < n:
                base = int(cs[i - 1]) if i else 0
                j = int(np.searchsorted(cs, base + bins, side="right"))
                j = min(n, max(j, i + m))
                batches.append(order[s + i : s + j])
                i = j
        return batches

    def _build(self, epoch: int) -> None:
        batches = self._cut()
        if self.dataset.train:
            rng = np.random.default_rng([self.seed, epoch])
            batches = [batches[i] for i in rng.permutation(len(batches))]
        keep = len(batches) // self.world_size * self.world_size
        self._batches = batches[self.rank : keep : self.world_size]
        self._epoch = epoch

    def _current_epoch(self) -> int:
        return int(self.dataset.epoch) if self.dataset.train else 0

    def __iter__(self) -> Iterator[List[int]]:
        """Yield index lists for this rank and the dataset's current epoch."""
        epoch = self._current_epoch()
        if self._epoch != epoch:
            self._build(epoch)
        for b in self._batches:
            yield b.tolist()

    def __len__(self) -> int:
        """Return the number of batches for this rank."""
        epoch = self._current_epoch()
        if self._epoch != epoch:
            self._build(epoch)
        return len(self._batches)
