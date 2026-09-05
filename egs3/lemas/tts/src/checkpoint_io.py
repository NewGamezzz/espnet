"""Checkpoint IO that memory-maps checkpoints on load.

Plain ``torch.load()`` materializes the full checkpoint on every rank's
heap during resume (6.9 GB per rank for this recipe). glibc does not
return those freed pages to the OS, leaving several GB of dead resident
memory per rank. On nodes whose page reclaim is already under pressure,
that pinned memory pushes the job past the threshold where the kernel
starts evicting the ranks' own memory-mapped library pages, and the
forward pass livelocks on filesystem page faults (2026-07-28 Delta
resume-stall investigation: unfixed resumes died at batch 450-1050 on
four nodes; with this IO the same resume ran 6800 batches at fresh-run
speed, rank RSS 4.0 GB instead of 9.5 GB).

``mmap=True`` keeps the checkpoint file-backed: ``load_state_dict``
copies values into existing storages and the mapping is dropped
afterwards, so the heap spike never happens.
"""

import os
from typing import Any, Optional

import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO


class MmapCheckpointIO(TorchCheckpointIO):
    """TorchCheckpointIO whose load path uses ``torch.load(..., mmap=True)``.

    The signature mirrors ``TorchCheckpointIO.load_checkpoint`` in
    Lightning 2.6 (``path``, ``map_location``, ``weights_only``) so the
    checkpoint connector can call it with any combination of those
    arguments. Non-local paths (fsspec URLs) fall back to the parent
    implementation, which handles remote filesystems but cannot mmap.
    """

    def load_checkpoint(
        self,
        path: Any,
        map_location: Optional[Any] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
    ) -> dict:
        """Load a checkpoint, memory-mapping local files."""
        if isinstance(path, (str, os.PathLike)) and os.path.isfile(path):
            return torch.load(
                path,
                map_location=map_location,
                mmap=True,
                # Lightning checkpoints carry non-tensor objects (loops,
                # callbacks, hyperparameters); a None from the connector
                # must mean "trusted own checkpoint", not torch's default.
                weights_only=False if weights_only is None else weights_only,
            )
        return super().load_checkpoint(path, map_location, weights_only=weights_only)
