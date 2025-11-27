# This script is copied from
# https://github.com/Lightning-AI/pytorch-lightning/issues/3346#issuecomment-1478556073

from typing import Any

import torch
from typeguard import typechecked

from espnet3.components.multiple_optim import MultipleOptim


class MultipleScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Wrap a scheduler so Lightning accepts the ``MultipleOptim`` wrapper.

    Lightning expects a scheduler to expose an ``optimizer`` attribute. This shim
    forwards everything to the wrapped scheduler while reporting the
    ``MultipleOptim`` instance. It allows one scheduler per optimizer inside
    ``MultipleOptim`` without triggering Lightning's per-optimizer training_step
    logic.

    Args:
        multiple_optimizer (MultipleOptim): The wrapper that owns the underlying
            optimizers.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler configured
            for a single optimizer inside ``multiple_optimizer``.
        optimizer_idx (int): Index of the optimizer the scheduler controls.

    Note:
        Only minimal delegation is implemented. Any attribute not explicitly
        intercepted is passed through to the wrapped scheduler.

    Example:
        >>> ms = MultipleScheduler(multi_optim, StepLR(opt_a, step_size=10), 0)
        >>> lr = ms.get_last_lr()
    """

    @typechecked
    def __init__(
        self,
        multiple_optimizer: MultipleOptim,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        optimizer_idx: int,
    ) -> None:
        assert 0 <= optimizer_idx < len(multiple_optimizer.optimizers), (
            f"optimizer_idx {optimizer_idx} is out of range for "
            f"multiple_optimizer with {len(multiple_optimizer.optimizers)} optimizers."
        )
        self.optimizer = multiple_optimizer
        self.lr_scheduler = lr_scheduler
        self.idx = optimizer_idx

    def __getattr__(self, name: str) -> Any:
        if name in {"optimizer", "lr_scheduler", "idx"}:
            return getattr(self, name)
        else:
            return self.lr_scheduler.__getattribute__(name)
