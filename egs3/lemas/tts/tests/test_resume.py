"""Resume semantics the training chain relies on, against the installed Lightning.

A chain link is killed by its walltime in the middle of an epoch and the next
link resumes from ``last.ckpt``, written at the end of the last finished
epoch. The resume is exact only when a validation is due at the end of that
epoch: Lightning 2.6.5 otherwise trains the first batch of the finished epoch
once more and takes an optimizer step on it (measured on Delta 2026-09-29,
smoke jobs 22548970/22548971, with validation every 5th epoch). The test runs
with the validation cadence of the production config.
"""

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from src.sampler import BlockBatchSampler
from tests.test_dataset import _ds

PER_EPOCH, ACCUMULATE = 2, 2  # one optimizer step per epoch


class _Killed(Exception):
    pass


class _Rows(torch.utils.data.Dataset):
    """Row indexes only: the batches are what the test looks at."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([float(i)])


def _sampler(ds, epoch):
    ds.set_epoch(epoch)
    return BlockBatchSampler(
        ds, batch_bins=100 * 300, seed=1, batches_per_epoch=PER_EPOCH
    )


class _Module(L.LightningModule):
    def __init__(self, ds, kill_in_epoch=None):
        super().__init__()
        self.w = torch.nn.Linear(1, 1)
        self.ds, self.kill_in_epoch, self.events = ds, kill_in_epoch, []

    def training_step(self, batch, batch_idx):
        rows = tuple(int(i) for i in batch.flatten().tolist())
        self.events.append((self.current_epoch, self.global_step, rows))
        if self.current_epoch == self.kill_in_epoch and batch_idx == 1:
            raise _Killed  # the walltime, in the middle of an epoch
        return self.w(batch).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.log("valid/loss", self.w(batch).pow(2).mean())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            _Rows(len(self.ds)), batch_sampler=_sampler(self.ds, self.current_epoch)
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(_Rows(4), batch_size=2)


def _trainer(exp, max_steps):
    cfg = OmegaConf.load("conf/training_f5_base_dualprompt.yaml").trainer
    last = ModelCheckpoint(  # as espnet3's get_default_callbacks builds it
        dirpath=exp,
        save_last="link",
        filename="step{step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )
    return L.Trainer(
        max_steps=max_steps,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        accumulate_grad_batches=ACCUMULATE,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[last],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_a_killed_link_is_resumed_at_the_next_epoch_without_replay(corpus, tmp_path):
    ds = _ds(corpus, block_samples=32000)
    exp = tmp_path / "exp"
    first = _Module(ds, kill_in_epoch=2)
    with pytest.raises(_Killed):
        _trainer(exp, max_steps=50).fit(first)
    assert (exp / "last.ckpt").resolve().name == "step2.ckpt"  # end of epoch 1

    second = _Module(ds)
    _trainer(exp, max_steps=4).fit(second, ckpt_path=str(exp / "last.ckpt"))

    expected = [
        (epoch, epoch, tuple(batch))  # one step per epoch: global_step == epoch
        for epoch in (2, 3)
        for batch in _sampler(ds, epoch)
    ]
    assert second.events == expected
    assert (exp / "last.ckpt").resolve().name == "step4.ckpt"
