# f5tts/callbacks/ema_callback.py

from __future__ import annotations

import torch
import torch.distributed as dist
import lightning.pytorch as pl
from ema_pytorch import EMA


class EMACallback(pl.Callback):
    """
    Ports F5-TTS EMA behavior into ESPnet3's PL callback system.

    - Updates once per true optimizer step (not per micro-step).
    - Swaps EMA weights in for validation/test, restores afterward.
    - Saves under 'ema_model_state_dict' matching F5-TTS checkpoint format.
    """

    def __init__(self, decay: float = 0.9999, warmup_steps: int = 0, **ema_kwargs):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.ema_kwargs = ema_kwargs
        self.ema: EMA | None = None
        self._backup: dict | None = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        # Initialize EMA on the main process only, matching F5-TTS
        # (`if self.is_main: self.ema_model = EMA(...)`). On other ranks self.ema
        # stays None, so update / save / validation-swap all no-op there (each is
        # guarded by `self.ema is not None`).
        if stage == "fit" and trainer.is_global_zero:
            self.ema = EMA(
                pl_module.model,
                beta=self.decay,
                include_online_model=False,
                **self.ema_kwargs,
            ).to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Mirror F5-TTS exactly:
        #   if accelerator.sync_gradients:   # a real optimizer update (end of accum)
        #       if is_main:                  # main process only
        #           ema_model.update()
        # on_train_batch_end runs AFTER optimizer.step()/scheduler.step()/zero_grad(),
        # so this is the post-zero_grad, once-per-update, main-only placement.
        if self.ema is None or not trainer.is_global_zero:
            return
        accum = max(int(getattr(trainer, "accumulate_grad_batches", 1)), 1)
        if (batch_idx + 1) % accum != 0:  # not a true optimizer step yet
            return
        if trainer.global_step >= self.warmup_steps:
            self.ema.update()

    def _swap_in_ema(self, trainer, pl_module):
        """Load the EMA weights into the online model for evaluation.

        ``ema_pytorch.EMA`` keeps the averaged weights in ``self.ema.ema_model``
        (a copy of the online model) and has no torch-ema-style store/restore, so
        we back up the online weights and load ``ema_model``'s state.

        EMA lives only on the main rank (``self.ema`` is None elsewhere). On
        multi-GPU we therefore broadcast the main rank's EMA weights to every rank
        so all ranks validate on the same EMA weights (otherwise non-main ranks
        would validate on the online model and pollute the aggregated metric).
        """
        distributed = (
            trainer.world_size > 1 and dist.is_available() and dist.is_initialized()
        )

        # Only the main rank knows whether EMA exists; agree across ranks so all
        # ranks take the same path (and issue the same collectives).
        has_ema = self.ema is not None
        if distributed:
            flag = torch.tensor([int(has_ema)], device=pl_module.device)
            dist.broadcast(flag, src=0)
            has_ema = bool(flag.item())
        if not has_ema:
            return

        self._backup = {
            k: v.detach().clone() for k, v in pl_module.model.state_dict().items()
        }
        if self.ema is not None:  # main rank: put EMA weights into the online model
            pl_module.model.load_state_dict(self.ema.ema_model.state_dict())
        if distributed:  # send main's (EMA) weights to every rank, in registration order
            for tensor in pl_module.model.state_dict().values():
                if torch.is_tensor(tensor):
                    dist.broadcast(tensor, src=0)

    def _restore_online(self, pl_module):
        """Restore the online weights saved by :meth:`_swap_in_ema`.

        Restore is keyed on ``self._backup`` (set on every rank that swapped), not
        on ``self.ema`` — non-main ranks have ``self.ema is None`` but still hold a
        backup, and must restore to avoid leaving EMA weights in the online model
        (which would desync DDP when training resumes).
        """
        if self._backup is None:
            return
        pl_module.model.load_state_dict(self._backup)
        self._backup = None

    def on_validation_start(self, trainer, pl_module):
        self._swap_in_ema(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._restore_online(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_in_ema(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        self._restore_online(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.is_global_zero and self.ema is not None:
            checkpoint["ema_model_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_model_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_model_state_dict"])