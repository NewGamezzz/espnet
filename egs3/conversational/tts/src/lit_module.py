"""Recipe LightningModule: two-group optimizer + packed conversation batches.

Subclasses ``ESPnetLightningModule`` for exactly two reasons, both outside
what the stock espnet3 config paths can express:

- ``configure_optimizers``: ONE optimizer with two param groups (injected
  exchange modules at ``optim.lr_exchange``, pretrained backbone at
  ``optim.lr_backbone``) so the communication modules can train faster than
  the trunk.  The espnet3 single-optimizer path flattens all parameters
  into one group; the named multi-optimizer path forces manual optimization
  and cannot select "everything except the exchanges".
- ``train_dataloader``/``val_dataloader``: a standard ``DataLoader`` around
  ``ConversationBatchSampler`` built fresh each epoch (the espnet3 trainer
  forces ``reload_dataloaders_every_n_epochs=1``) with the CURRENT epoch,
  giving seeded per-epoch batch reshuffling.  The espnet3 iter_factory path
  cannot do this: ``DataLoaderBuilder._build_iter_factory`` hardcodes
  ``build_iter(epoch, shuffle=False)``, freezing the batch order across
  epochs even when the config says ``shuffle: true``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egs3.conversational.tts.dataset import collate_conversations
from egs3.conversational.tts.src.build_model import exchange_param_groups
from egs3.conversational.tts.src.sampler import ConversationBatchSampler
from espnet3.components.modeling.lightning_module import ESPnetLightningModule

logger = logging.getLogger("lightning")


class PackedConversationCollator:
    """``collate_conversations`` in the ``(ids, batch_dict)`` trainer contract.

    ``ESPnetLightningModule._step`` calls ``model(**batch[1])``, so the
    collator returns the window ids first and the packed tensors second.
    """

    def __init__(self, text_pad_value: int | None = None):
        self.text_pad_value = text_pad_value

    def __call__(self, samples: list[dict[str, Any]]):
        batch = collate_conversations(samples, text_pad_value=self.text_pad_value)
        window_ids = batch.pop("window_ids")
        return window_ids, batch


class ConversationalLightningModule(ESPnetLightningModule):
    """ESPnetLightningModule with exchange/backbone param groups and the
    conversation batch sampler (see module docstring)."""

    def _log_stats(self, mode, stats, weight, extra_stats=None):
        """Route per-channel loss means around the ``sync_dist`` path.

        ``loss_ch{k}`` keys vary with the batch's largest channel count, and
        the parent logs validation stats with ``sync_dist=True`` (one
        all-reduce per metric): with mixed-N batches, two DDP ranks can log
        different key sets in the same step - mismatched collectives, i.e. an
        NCCL deadlock.  The per-channel means stay logged (under the random
        channel permutation they are symmetric in expectation, so a
        ch0/ch1 split during training is a row-symmetry bug canary), just
        never synced.
        """
        stats = dict(stats)
        per_channel = {k: stats.pop(k) for k in list(stats) if k.startswith("loss_ch")}
        super()._log_stats(mode, stats, weight, extra_stats)
        if per_channel and getattr(self, "_trainer", None) is not None:
            self.log_dict(
                {
                    f"{mode}/{k}": v.item() if hasattr(v, "item") else v
                    for k, v in per_channel.items()
                },
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

    def configure_optimizers(self):
        """One optimizer over two param groups + the house-style scheduler.

        Config contract (replaces the stock ``optimizer:`` block)::

            optim:
              _target_: torch.optim.AdamW   # first positional arg = groups
              lr_exchange: 1.0e-4
              lr_backbone: 1.0e-5
              betas: [0.9, 0.999]
              weight_decay: 0.01
            scheduler: ...                  # instantiated with optimizer=
            scheduler_interval: step
        """
        optim_config = OmegaConf.to_container(self.config.optim, resolve=True)
        lr_exchange = float(optim_config.pop("lr_exchange"))
        lr_backbone = float(optim_config.pop("lr_backbone"))
        optim_config.setdefault("_target_", "torch.optim.AdamW")
        groups = exchange_param_groups(self.model, lr_exchange, lr_backbone)
        optimizer = instantiate(optim_config, groups)

        scheduler = instantiate(
            OmegaConf.to_container(self.config.scheduler, resolve=True),
            optimizer=optimizer,
        )
        interval = str(getattr(self.config, "scheduler_interval", "step"))
        if interval not in {"step", "epoch"}:
            raise AssertionError("scheduler_interval must be 'step' or 'epoch'")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval},
        }

    def _packed_dataloader(self, dataset, mode: str) -> torch.utils.data.DataLoader:
        dataset.use_espnet_collator = False  # collator takes raw sample dicts
        loader_config = OmegaConf.to_container(
            getattr(self.config.dataloader, mode), resolve=True
        )
        sampler = ConversationBatchSampler(
            dataset,
            batch_bins=int(loader_config.pop("batch_bins")),
            min_batch_size=int(loader_config.pop("min_batch_size", 1)),
            shuffle=bool(loader_config.pop("shuffle", mode == "train")),
            seed=int(self.config.get("seed") or 0),
            epoch=self.current_epoch,
        )
        logger.info(
            "[%s] ConversationBatchSampler: %d batches (epoch=%d)",
            mode,
            len(sampler),
            self.current_epoch,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            **loader_config,
        )

    def train_dataloader(self):
        return self._packed_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._packed_dataloader(self.valid_dataset, "valid")
