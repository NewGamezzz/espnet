"""Non-invasive injection of exchange modules into an existing backbone.

``inject_exchange`` replaces selected entries of the backbone's block
``nn.ModuleList`` with ``ExchangedBlock`` wrappers; ``remove_exchange``
restores the original blocks (and the original state-dict keys) exactly.

The branch axis is folded into the batch dimension: the wrapped model is
called with hidden states of shape ``(B*N, T, d)`` and each wrapper unfolds to
``(B, N, T, d)`` for the exchange, guided by a shared ``BranchContext``.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .registry import BlockSpec
from .schedule import ExchangeSchedule, Mode

_NO_SNAPSHOT = object()

# Same signal torch.utils.checkpoint uses to key recomputation: user code only
# runs inside a graph task while backward re-executes a checkpointed region.
_current_graph_task_id = getattr(torch._C, "_current_graph_task_id", lambda: -1)


class BranchContext:
    """Runtime switch telling every ``ExchangedBlock`` how many branches are
    folded into the batch dimension.

    Plain object, NOT an ``nn.Module``. Inactive (``n_branch is None``) means
    the wrapped blocks behave exactly like the originals.
    """

    def __init__(self):
        self.n_branch: int | None = None
        self.pad_mask: torch.Tensor | None = None

    @property
    def active(self) -> bool:
        return self.n_branch is not None

    @contextmanager
    def branches(self, n: int, pad_mask: torch.Tensor | None = None):
        """Activate exchanges for ``n`` folded branches; exception-safe.

        With activation checkpointing (``use_reentrant=False``), ``backward()``
        must also run inside this context: the recompute re-executes each
        ``ExchangedBlock.forward``, which verifies the context still matches
        the one seen at forward time and raises ``RuntimeError`` otherwise.
        """
        self.n_branch = n
        self.pad_mask = pad_mask
        try:
            yield self
        finally:
            self.n_branch = None
            self.pad_mask = None


class ExchangedBlock(nn.Module):
    """Wraps one transformer block and applies an exchange to its hidden output.

    ``base_block`` and ``exchange`` are registered submodules; ``ctx`` and
    ``spec`` are plain attributes (set via ``object.__setattr__``) so they are
    never registered as submodules and never appear in the state dict.

    With an active context, the block output's hidden tensor ``(B*N, T, d)`` is
    unfolded to ``(B, N, T, d)``, exchanged across the branch axis, and folded
    back. ``unflatten(0, (-1, n_branch))`` stays correct under CFG-style
    batches (cond and uncond concatenated along batch) because each segment's
    length is a multiple of ``n_branch``, so groups never straddle the
    segment boundary.

    The context is read at call time, so under activation checkpointing
    (``use_reentrant=False``) ``backward()`` must run inside the same
    ``ctx.branches(...)`` context as the forward: each forward snapshots the
    context state, and the checkpoint recompute raises ``RuntimeError`` if the
    context has since been exited or changed, instead of silently skipping the
    exchange and producing wrong gradients.
    """

    def __init__(
        self,
        base_block: nn.Module,
        exchange: nn.Module,
        ctx: BranchContext,
        spec: BlockSpec,
    ):
        super().__init__()
        self.base_block = base_block
        self.exchange = exchange
        object.__setattr__(self, "ctx", ctx)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "_fwd_snapshot", _NO_SNAPSHOT)

    def _validate_recompute(self):
        snapshot = self._fwd_snapshot
        if snapshot is _NO_SNAPSHOT:
            return
        n_branch, pad_mask = snapshot
        if self.ctx.n_branch != n_branch or self.ctx.pad_mask is not pad_mask:
            raise RuntimeError(
                "BranchContext changed between the checkpointed forward and its "
                f"recomputation during backward: forward saw n_branch={n_branch}, "
                f"recompute sees n_branch={self.ctx.n_branch}. With activation "
                "checkpointing, backward() must be called inside the same "
                "ctx.branches(...) context as the forward pass."
            )

    def forward(self, *args, **kwargs):
        if _current_graph_task_id() != -1:
            self._validate_recompute()
        else:
            object.__setattr__(
                self, "_fwd_snapshot", (self.ctx.n_branch, self.ctx.pad_mask)
            )
        out = self.base_block(*args, **kwargs)
        if not self.ctx.active:
            return out
        h = self.spec.unpack(out)  # (B*N, T, d)
        h = h.unflatten(0, (-1, self.ctx.n_branch))  # (B, N, T, d)
        h = self.exchange(h, pad_mask=self.ctx.pad_mask)
        return self.spec.repack(out, h.flatten(0, 1))


def _resolve_blocks(model: nn.Module, path: str) -> nn.ModuleList:
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(
            f"{path!r} on {type(model).__name__} is {type(obj).__name__}, "
            "expected nn.ModuleList"
        )
    return obj


def inject_exchange(
    model: nn.Module,
    spec: BlockSpec,
    schedule: ExchangeSchedule,
    ctx: BranchContext,
) -> nn.Module:
    """Replace ``blocks[i]`` with an ``ExchangedBlock`` for every ``P_TAC``
    depth in ``schedule``; ``P`` blocks stay untouched (original object,
    original state-dict keys). Modifies ``model`` in place and returns it.
    """
    blocks = _resolve_blocks(model, spec.path)
    if schedule.depth != len(blocks):
        raise ValueError(
            f"schedule depth {schedule.depth} != number of blocks {len(blocks)}"
        )
    for i in range(len(blocks)):
        if schedule.mode(i) is Mode.P_TAC:
            if isinstance(blocks[i], ExchangedBlock):
                raise ValueError(
                    f"block {i} is already an ExchangedBlock; remove_exchange first"
                )
            blocks[i] = ExchangedBlock(blocks[i], schedule.exchange_for(i), ctx, spec)
    return model


def remove_exchange(model: nn.Module, spec: BlockSpec) -> nn.Module:
    """Restore every ``ExchangedBlock`` back to its ``base_block``.

    Modifies ``model`` in place and returns it; the resulting state dict has
    exactly the original keys and values.
    """
    blocks = _resolve_blocks(model, spec.path)
    for i in range(len(blocks)):
        if isinstance(blocks[i], ExchangedBlock):
            blocks[i] = blocks[i].base_block
    return model
