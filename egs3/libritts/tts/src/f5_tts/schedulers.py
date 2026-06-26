"""Learning-rate schedulers for the F5-TTS recipe.

``linear_warmup_decay`` reproduces F5-TTS's training schedule
(https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/trainer.py): a
linear warmup ``1e-8 -> 1.0`` over ``warmup_steps`` followed by a linear decay
``1.0 -> 1e-8`` over the remaining steps, chained with ``SequentialLR``. Stepped
per optimizer update (``scheduler_interval: step``).

Unlike ESPnet's ``WarmupLR`` (inverse-sqrt, non-terminating), the linear decay
needs to know the total number of updates, so ``total_steps`` must be set to the
planned training length (``epochs * updates_per_epoch``), matching how upstream
computes ``decay_updates = total_updates - warmup_updates``.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import LinearLR, SequentialLR


def linear_warmup_decay(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    start_factor: float = 1e-8,
    end_factor: float = 1e-8,
):
    """LinearLR warmup then LinearLR decay via SequentialLR (F5-TTS schedule)."""
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    decay_steps = max(total_steps - warmup_steps, 1)

    warmup = LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )
    decay = LinearLR(
        optimizer, start_factor=1.0, end_factor=end_factor, total_iters=decay_steps
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
    )
