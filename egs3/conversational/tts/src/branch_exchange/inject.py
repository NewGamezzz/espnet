"""Non-invasive injection of exchange modules into an existing backbone.

``inject_exchange`` replaces selected entries of the backbone's block
``nn.ModuleList`` with ``ExchangedBlock`` wrappers; ``remove_exchange``
restores the original blocks (and the original state-dict keys) exactly.

The branch axis is folded into the batch dimension as a packed layout:
``ctx.branches(counts=[n_1, ..., n_B])`` declares that the model is called
with ``(M, T, d)`` hidden states where ``M = sum(n_i)`` stacks every
conversation's branches with NO padding rows; each wrapper hands the packed
hidden states plus per-row conversation ids to its exchange.
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
    """Runtime switch telling every ``ExchangedBlock`` how the rows of the
    packed batch group into conversations.

    Plain object, NOT an ``nn.Module``. Inactive (``conv_id is None``) means
    the wrapped blocks behave exactly like the originals.
    """

    def __init__(self):
        self.conv_id: torch.Tensor | None = None
        self.n_conv: int | None = None
        self._segment_cache: dict = {}

    @property
    def active(self) -> bool:
        return self.conv_id is not None

    @contextmanager
    def branches(self, counts, device=None):
        """Activate exchanges for the enclosed forward/backward; exception-safe.

        ``counts`` lists each conversation's branch count; the model input
        stacks all branches as ``(sum(counts), T, d)`` with no padding.
        ``device`` places the derived per-row conversation ids (default CPU;
        they are moved and cached per device on first use).

        With activation checkpointing (``use_reentrant=False``), ``backward()``
        must also run inside this context: the recompute re-executes each
        ``ExchangedBlock.forward``, which verifies the context still matches
        the one seen at forward time and raises ``RuntimeError`` otherwise.
        """
        counts_t = torch.as_tensor(counts, dtype=torch.long)
        if counts_t.ndim != 1 or counts_t.numel() == 0 or bool((counts_t < 1).any()):
            raise ValueError(f"counts must be a non-empty sequence of positive ints, got {counts!r}")
        self.conv_id = torch.repeat_interleave(
            torch.arange(counts_t.numel(), dtype=torch.long), counts_t
        ).to(device)
        self.n_conv = int(counts_t.numel())
        try:
            yield self
        finally:
            self.conv_id = None
            self.n_conv = None
            self._segment_cache.clear()

    def segment_conv_id(self, m: int, device) -> tuple[torch.Tensor, int]:
        """Per-row conversation ids for a packed hidden batch of ``m`` rows.

        ``m`` may be any positive multiple of the base total (CFG-style
        inference concatenates cond and uncond along the batch, repeating the
        packed layout per segment); each segment's ids get an ``n_conv``
        offset so branches never mix across segments. Results are cached per
        (segment count, device) for the lifetime of the context.
        """
        total = int(self.conv_id.shape[0])
        if m % total != 0:
            raise RuntimeError(
                f"packed hidden batch has {m} rows, not a multiple of the "
                f"context total {total} (= sum of counts); the model input "
                "must stack whole copies of the packed layout"
            )
        segments = m // total
        key = (segments, device)
        cached = self._segment_cache.get(key)
        if cached is None:
            conv_id = self.conv_id.to(device)
            if segments > 1:
                offsets = torch.arange(segments, dtype=torch.long, device=device) * self.n_conv
                conv_id = (conv_id.unsqueeze(0) + offsets[:, None]).reshape(-1)
            cached = (conv_id, segments * self.n_conv)
            self._segment_cache[key] = cached
        return cached


class ExchangedBlock(nn.Module):
    """Wraps one transformer block and applies an exchange to its hidden output.

    ``base_block`` and ``exchange`` are registered submodules; ``ctx`` and
    ``spec`` are plain attributes (set via ``object.__setattr__``) so they are
    never registered as submodules and never appear in the state dict.

    With an active context, the block output's hidden tensor ``(M, T, d)`` is
    handed to the exchange together with per-row conversation ids from the
    context. CFG-style batches (cond and uncond concatenated along batch)
    repeat the packed layout per segment and get offset ids, so branches
    never mix across segments.

    The context is read at call time, so under activation checkpointing
    (``use_reentrant=False``) ``backward()`` must run inside the same
    ``ctx.branches(...)`` context as the forward: each forward snapshots the
    context state, and the checkpoint recompute raises ``RuntimeError`` if the
    context has since been exited or changed, instead of silently skipping the
    exchange and producing wrong gradients.
    """

    def __init__(self, base_block: nn.Module, exchange: nn.Module, ctx: BranchContext, spec: BlockSpec):
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
        if self.ctx.conv_id is not snapshot:
            raise RuntimeError(
                "BranchContext changed between the checkpointed forward and its "
                "recomputation during backward. With activation checkpointing, "
                "backward() must be called inside the same ctx.branches(...) "
                "context as the forward pass."
            )

    def forward(self, *args, **kwargs):
        if _current_graph_task_id() != -1:
            self._validate_recompute()
        else:
            object.__setattr__(self, "_fwd_snapshot", self.ctx.conv_id)
        out = self.base_block(*args, **kwargs)
        if not self.ctx.active:
            return out
        h = self.spec.unpack(out)  # (M, T, d)
        conv_id, n_conv = self.ctx.segment_conv_id(h.shape[0], h.device)
        return self.spec.repack(out, self.exchange(h, conv_id, n_conv=n_conv))


def _resolve_blocks(model: nn.Module, path: str) -> nn.ModuleList:
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"{path!r} on {type(model).__name__} is {type(obj).__name__}, expected nn.ModuleList")
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
        raise ValueError(f"schedule depth {schedule.depth} != number of blocks {len(blocks)}")
    for i in range(len(blocks)):
        if schedule.mode(i) is Mode.P_TAC:
            if isinstance(blocks[i], ExchangedBlock):
                raise ValueError(f"block {i} is already an ExchangedBlock; remove_exchange first")
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
