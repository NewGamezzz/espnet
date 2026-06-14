# f5tts/callbacks/ema_callback.py

from __future__ import annotations
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

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if stage == "fit":
            self.ema = EMA(
                pl_module.model,
                beta=self.decay,
                include_online_model=False,
                **self.ema_kwargs,
            ).to(pl_module.device)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        # Update EMA *after* the optimizer step (Lightning calls this hook after
        # optimizer.step(), before zero_grad), once per true optimizer update —
        # matching F5-TTS, which calls ema_model.update() after optimizer.step().
        # Only update ema on the global zero process to avoid redundant updates 
        # across distributed workers.
        if trainer.is_global_zero and self.ema is not None:
            if trainer.global_step >= self.warmup_steps:
                self.ema.update()

    def on_validation_start(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_params_from_ema_to_model()

    def on_validation_end(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.restore()

    def on_test_start(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_params_from_ema_to_model()

    def on_test_end(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.restore()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.is_global_zero and self.ema is not None:
            checkpoint["ema_model_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_model_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_model_state_dict"])