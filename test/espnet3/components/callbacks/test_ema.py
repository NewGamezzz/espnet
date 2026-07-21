import lightning.pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from espnet3.components.callbacks.ema import EMACallback

# ===============================================================
# Test Case Summary for EMACallback
# ===============================================================
#
# Normal Cases
# | Test Name                                      | Description                       |
# |------------------------------------------------|-----------------------------------|
# | test_ema_updates_once_per_optimizer_step       | EMA updates exactly once per real |
# |                                                | optimizer step, including the     |
# |                                                | epoch-final partial accumulation  |
# |                                                | window (7 batches, accum=2).      |
# | test_ema_updates_without_accumulation          | Sanity: with accum=1 every batch  |
# |                                                | is an optimizer step.             |


class _TinyModule(pl.LightningModule):
    """Minimal module exposing ``self.model`` as EMACallback expects."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2, 2)

    def training_step(self, batch, batch_idx):
        (x,) = batch
        return self.model(x).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)


def _fit(n_samples: int, accumulate_grad_batches: int, max_epochs: int = 1):
    torch.manual_seed(0)
    module = _TinyModule()
    loader = DataLoader(TensorDataset(torch.randn(n_samples, 2)), batch_size=1)
    callback = EMACallback(decay=0.9999)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, loader)
    return trainer, callback


def test_ema_updates_once_per_optimizer_step():
    """EMA must update on the epoch-final partial accumulation window.

    With 7 batches per epoch and ``accumulate_grad_batches=2``, Lightning
    steps the optimizer after batch indices 1, 3, 5, and 6 (it flushes the
    partial window at epoch end).  Upstream F5-TTS updates EMA on every
    ``accelerator.sync_gradients`` step, so the EMA step counter must equal
    ``trainer.global_step`` — 4 per epoch, not 3.
    """
    trainer, callback = _fit(n_samples=7, accumulate_grad_batches=2, max_epochs=2)
    assert trainer.global_step == 8  # 4 real optimizer steps per epoch
    assert callback.ema is not None
    assert int(callback.ema.step.item()) == trainer.global_step


def test_ema_updates_without_accumulation():
    trainer, callback = _fit(n_samples=3, accumulate_grad_batches=1)
    assert trainer.global_step == 3
    assert int(callback.ema.step.item()) == trainer.global_step
