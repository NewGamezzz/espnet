"""Lightning module that tells LEMASDataset the epoch before each loader build.

espnet3 rebuilds the dataloader every epoch (``DataLoaderBuilder`` receives
``epoch=self.current_epoch``) but never tells the dataset. The online prompt
draws are seeded by ``(seed, epoch, row)``, so the dataset must learn the
epoch before its workers fork; this override does that.
"""

from __future__ import annotations

from espnet3.components.modeling.lightning_module import ESPnetLightningModule


def _set_epoch(dataset, epoch: int) -> None:
    """Call ``set_epoch`` on ``dataset`` and on every wrapped dataset."""
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    for attr in ("datasets", "dataset"):
        inner = getattr(dataset, attr, None)
        if inner is None or inner is dataset:
            continue
        for d in inner if isinstance(inner, (list, tuple)) else [inner]:
            _set_epoch(d, epoch)


class LEMASLightningModule(ESPnetLightningModule):
    """ESPnetLightningModule with per-epoch dataset notification."""

    def train_dataloader(self):
        """Propagate the epoch, then build the loader as the parent does."""
        _set_epoch(self.train_dataset, self.current_epoch)
        return super().train_dataloader()

    def val_dataloader(self):
        """Propagate the epoch (a no-op for fixed validation draws)."""
        _set_epoch(self.valid_dataset, self.current_epoch)
        return super().val_dataloader()
