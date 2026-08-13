"""Recipe LightningModule: two-group optimizer + packed conversation batches.

Subclasses ``ESPnetLightningModule`` for three reasons, all outside what the
stock espnet3 config paths can express:

- ``configure_optimizers``: ONE optimizer with two param groups (injected
  exchange modules at ``optim.lr_exchange``, pretrained backbone at
  ``optim.lr_backbone``) so the communication modules can train faster than
  the trunk.  The espnet3 single-optimizer path flattens all parameters
  into one group; the named multi-optimizer path forces manual optimization
  and cannot select "everything except the exchanges".
- ``train_dataloader``/``val_dataloader``: a standard ``DataLoader`` around
  ``ConversationBatchSampler`` built once per fit.  Per-epoch batch
  reshuffling is delivered by ``ConversationBatchSampler.set_epoch``, which
  Lightning calls at the start of every epoch - except the first one after a
  resume: ``FitLoop.run`` materializes the dataloader iterator (eagerly
  consumed by worker prefetch) before ``set_epoch`` runs, so on resume the
  constructor's ``epoch=`` value is what actually seeds the first epoch's
  order, not a value ``set_epoch`` will still get to correct.  The
  constructor therefore seeds the sampler from the fit loop's
  processed-epoch count (``trainer.fit_loop.epoch_progress.current.processed``)
  rather than ``self.current_epoch`` (which reads the completed-epoch count
  espnet3's ``save_last`` checkpoint stores as one epoch behind at
  resume time): that keeps the eagerly-materialized first iterator correct,
  and ``set_epoch`` keeps every subsequent epoch correct.  The
  espnet3 iter_factory path cannot do this: ``DataLoaderBuilder._build_iter_factory``
  hardcodes ``build_iter(epoch, shuffle=False)``, freezing the batch order
  across epochs even when the config says ``shuffle: true``.
- ``on_validation_end``: gated on ``trainer.sanity_checking``, releases the
  sanity probe's abandoned val dataloader iterator so its worker processes
  are shut down immediately instead of lingering until the first real
  validation (``on_sanity_check_end`` is Callback-only in Lightning 2.6.5,
  not a LightningModule hook, so it never fires here).
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


class PlannedWindowView(torch.utils.data.Dataset):
    """Route ``(component_idx, WindowRecord)`` specs from
    ``ConversationBatchSampler`` to the right ``ConversationDataset``, then
    apply that component's transform and preprocessor exactly like espnet3's
    ``CombinedDataset.__getitem__`` does for int indices.

    ``CombinedDataset`` caches ``cumulative_lengths`` at construction and only
    routes ints (and utterance-id strings) through them, so per-epoch plans -
    which have no fixed global index, only a ``(component_idx, record)`` pair
    - cannot flow through its integer indexing.  The sampler's specs are
    self-contained instead: no shared mutable plan for a ``Dataset`` to read
    out from under a re-planned epoch, no cross-worker-fork hazard.

    Accepts either espnet3's ``CombinedDataset`` (``.datasets``/``.transforms``
    present) or a bare ``ConversationDataset`` (neither present, so the
    transform/preprocessor step is a no-op and ``load_window`` output is
    returned as-is - matching how a single-component config never routes
    through ``CombinedDataset`` either).
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.datasets = getattr(dataset, "datasets", None) or [dataset]
        self.transforms = getattr(dataset, "transforms", None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, spec):
        comp_idx, record = spec
        sample = self.datasets[comp_idx].load_window(record)
        if self.transforms is not None:
            transform, preprocessor = self.transforms[comp_idx]
            sample = transform(sample)
            if getattr(self.dataset, "use_espnet_preprocessor", False):
                sample = preprocessor(record.window_id, sample)
            else:
                sample = preprocessor(sample)
        if getattr(self.dataset, "use_espnet_collator", False):
            return record.window_id, sample
        return sample


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

    def __init__(self, model, config):
        super().__init__(model, config)
        # ConversationBatchSampler strides batches by rank itself (and is not
        # a torch BatchSampler subclass), so the espnet3 trainer must pass
        # use_distributed_sampler=False to Lightning; it keys that off this
        # flag, which the parent only sets for iter_factory configs.
        self.is_espnet_sampler = True

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

    def _initial_epoch(self) -> int:
        """The epoch that should seed a freshly constructed sampler.

        On a normal (non-resume) fit this is ``self.current_epoch`` (0).  On
        resume, ``FitLoop.run`` materializes the train dataloader iterator
        before Lightning's ``set_epoch`` propagation runs (see module
        docstring), so the constructor value - not a later ``set_epoch`` call
        - determines the first resumed epoch's batch order.  The fit loop's
        processed-epoch count is the correct value there; ``self.current_epoch``
        still reads the completed-epoch count at that point, which espnet3's
        ``save_last`` checkpoint stores one epoch behind.
        """
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            return int(trainer.fit_loop.epoch_progress.current.processed)
        return self.current_epoch

    def _packed_dataloader(self, dataset, mode: str) -> torch.utils.data.DataLoader:
        dataset.use_espnet_collator = False  # collator takes raw sample dicts
        loader_config = OmegaConf.to_container(
            getattr(self.config.dataloader, mode), resolve=True
        )
        initial_epoch = self._initial_epoch()
        weights = loader_config.pop("weights", None)
        online = mode == "train"
        sampler = ConversationBatchSampler(
            dataset,
            batch_bins=int(loader_config.pop("batch_bins")),
            min_batch_size=int(loader_config.pop("min_batch_size", 1)),
            shuffle=bool(loader_config.pop("shuffle", mode == "train")),
            seed=int(self.config.get("seed") or 0),
            epoch=initial_epoch,
            weights=weights,
            online=online,
        )
        logger.info(
            "[%s] ConversationBatchSampler: %d batches (initial epoch=%d; "
            "online=%s; per-epoch reshuffle via set_epoch)",
            mode,
            len(sampler),
            initial_epoch,
            online,
        )
        return torch.utils.data.DataLoader(
            PlannedWindowView(dataset),
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            **loader_config,
        )

    def train_dataloader(self):
        return self._packed_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._packed_dataloader(self.valid_dataset, "valid")

    def on_train_start(self) -> None:
        super().on_train_start()
        # Return checkpoint-restore heap pages to the OS. Even with the
        # mmap-based checkpoint load (src/checkpoint_io.py), restore-time
        # allocations fragment the heap, and resident-anon bloat after a
        # resume is what pushed nodes past their page-reclaim threshold in
        # the 2026-07-28 resume-stall investigation.
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:  # non-glibc platforms (e.g. macOS dev boxes)
            pass

    def on_validation_end(self) -> None:
        """Shut down the sanity probe's dataloader workers.

        ``on_sanity_check_end`` is a Callback-only hook in Lightning 2.6.5 -
        removed from ``ModelHooks`` long ago (the changelog records
        "Deprecated the ``on_sanity_check_start`` hook in ``ModelHooks``");
        a plain ``hasattr(LightningModule, "on_sanity_check_end")`` is
        ``False`` and a live fit never calls it.  ``on_validation_end`` is
        the real ``ModelHooks`` substitute: it fires once per validation
        pass, including the sanity pass, while ``trainer.sanity_checking``
        is still ``True``.  The sanity check stops after
        ``num_sanity_val_steps`` of the valid batches, abandoning an
        unexhausted iterator whose worker processes otherwise linger until
        the first real validation rebuilds it (observed live: 16 idle
        workers for 30+ min on a 4-rank run).
        """
        super().on_validation_end()
        if self.trainer.sanity_checking:
            self._release_sanity_val_iterator()

    def _release_sanity_val_iterator(self) -> None:
        """``_DataFetcher.teardown()`` is Lightning's own worker-shutdown path:
        it resets its own state and then calls ``reset()`` on the SAME
        ``CombinedLoader`` object ``val_loop._combined_loader`` points at
        (``val_loop.reset()`` sets up the fetcher with that exact loader), so
        clearing the fetcher already clears the loader's iterator - no
        separate ``val_loop._combined_loader.reset()`` call is needed.
        """
        val_loop = self.trainer.fit_loop.epoch_loop.val_loop
        if val_loop._data_fetcher is not None:
            val_loop._data_fetcher.teardown()
            val_loop._data_fetcher = None
