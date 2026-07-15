"""Non-invasive injection of exchange modules into an existing backbone.

``inject_exchange`` replaces selected transformer blocks (located by the
spec's ``target``: dotted path, depth regex, or explicit name list - see
``BlockSpec``) with ``ExchangedBlock`` wrappers; ``remove_exchange`` restores
the original blocks (and the original state-dict keys) exactly.

The branch axis is folded into the batch dimension as a packed layout:
``ctx.branches(counts=[n_1, ..., n_B])`` declares that the model is called
with ``(M, T, d)`` hidden states where ``M = sum(n_i)`` stacks every
conversation's branches with NO padding rows; each wrapper hands the packed
hidden states plus per-row conversation ids to its exchange.

Optionally, ``ctx.branches(counts=..., align_offsets=..., align_len=...)``
restricts every exchange to a per-row aligned audio slab (each channel's text
prefix has a different length, so its audio start offset differs too): row
``i``'s exchange input is ``h[i, offsets[i] : offsets[i] + align_len]``, and
positions outside that slab pass through untouched. See ``_apply_aligned``
and ``BranchContext.segment_align_offsets``.
"""

from __future__ import annotations

import difflib
import re
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
        self.align_offsets: torch.Tensor | None = None
        self.align_len: int | None = None
        self._segment_cache: dict = {}

    @property
    def active(self) -> bool:
        return self.conv_id is not None

    @contextmanager
    def branches(self, counts, device=None, align_offsets=None, align_len=None):
        """Activate exchanges for the enclosed forward/backward; exception-safe.

        ``counts`` lists each conversation's branch count; the model input
        stacks all branches as ``(sum(counts), T, d)`` with no padding.
        ``device`` places the derived per-row conversation ids (default CPU;
        they are moved and cached per device on first use).

        ``align_offsets`` / ``align_len`` (optional, must be given together)
        restrict every exchange to a per-row aligned slab instead of the full
        sequence: row ``i``'s audio lives at ``[align_offsets[i],
        align_offsets[i] + align_len)`` (its text prefix length differs per
        channel, so the audio start offset does too); everything outside that
        slab passes through untouched. ``align_offsets`` must have shape
        ``(sum(counts),)`` - one offset per row, in the same row order as
        ``counts`` unrolls. See ``_apply_aligned``.

        With activation checkpointing (``use_reentrant=False``), ``backward()``
        must also run inside this context: the recompute re-executes each
        ``ExchangedBlock.forward``, which verifies the context still matches
        the one seen at forward time and raises ``RuntimeError`` otherwise.

        Not re-entrant: activating an already-active context would overwrite
        its state and the inner exit would clear what the outer block still
        needs, so nesting raises ``RuntimeError`` instead.
        """
        if self.conv_id is not None:
            raise RuntimeError("BranchContext is already active; ctx.branches(...) does not nest")
        counts_t = torch.as_tensor(counts, dtype=torch.long)
        if counts_t.ndim != 1 or counts_t.numel() == 0 or bool((counts_t < 1).any()):
            raise ValueError(f"counts must be a non-empty sequence of positive ints, got {counts!r}")
        if (align_offsets is None) != (align_len is None):
            raise ValueError("align_offsets and align_len must be given together")
        if align_offsets is not None:
            off = torch.as_tensor(align_offsets, dtype=torch.long)
            if off.ndim != 1 or off.shape[0] != int(counts_t.sum()):
                raise ValueError(
                    f"align_offsets must have shape (sum(counts),), got {tuple(off.shape)}"
                )
            if int(align_len) < 1 or bool((off < 0).any()):
                raise ValueError("align_len must be >= 1 and offsets non-negative")
            self.align_offsets = off.to(device)
            self.align_len = int(align_len)
        self.conv_id = torch.repeat_interleave(
            torch.arange(counts_t.numel(), dtype=torch.long), counts_t
        ).to(device)
        self.n_conv = int(counts_t.numel())
        try:
            yield self
        finally:
            self.conv_id = None
            self.n_conv = None
            self.align_offsets = None
            self.align_len = None
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

    def segment_align_offsets(self, m: int, device) -> torch.Tensor:
        """Per-row aligned-slab start offsets for a packed hidden batch of
        ``m`` rows, mirroring ``segment_conv_id``'s CFG-segment tiling.

        Unlike conversation ids (which get an ``n_conv`` offset per segment so
        branches never mix), the audio slab of segment ``s``'s row ``i`` is at
        the SAME relative offset as segment 0's row ``i`` - the CFG uncond
        copy repeats the identical packed layout, text-prefix lengths and
        all - so offsets simply TILE (``offsets.repeat(segments)``), not shift.
        Results are cached per (segment count, device) under a distinct cache
        key from ``segment_conv_id`` so the two caches never collide.
        """
        total = int(self.conv_id.shape[0])
        if m % total != 0:
            raise RuntimeError(
                f"packed hidden batch has {m} rows, not a multiple of the "
                f"context total {total} (= sum of counts); the model input "
                "must stack whole copies of the packed layout"
            )
        segments = m // total
        key = ("off", segments, device)
        cached = self._segment_cache.get(key)
        if cached is None:
            offsets = self.align_offsets.to(device)
            if segments > 1:
                offsets = offsets.repeat(segments)
            self._segment_cache[key] = offsets
            cached = offsets
        return cached


def _call_exchange(
    exchange: nn.Module, h: torch.Tensor, conv_id: torch.Tensor, n_conv: int
) -> torch.Tensor:
    """Run ``exchange`` in its own parameter dtype; return in ``h``'s dtype.

    ``inject_exchange`` builds exchanges via the caller's ``factory``, which
    constructs them in the default dtype (float32) regardless of the wrapped
    backbone's dtype, so a bf16-loaded backbone would otherwise crash on the
    first exchange ``Linear`` (``mat1 and mat2 must have the same dtype``).
    Casting the activations here - never the parameters - keeps the wrapper
    dtype-agnostic without mutating state at forward time (safe under DDP
    bucketing, ``torch.compile``, and checkpoint recompute). The zero-init
    identity stays bit-exact: bf16 -> float32 -> bf16 round-trips exactly,
    and at ``g == 0`` the exchange returns its input unchanged. When the
    dtypes already match (or the exchange has no parameters, e.g.
    ``IdentityExchange``), this is a plain call with no casts.
    """
    p = next(exchange.parameters(), None)
    if p is None or p.dtype == h.dtype:
        return exchange(h, conv_id, n_conv=n_conv)
    return exchange(h.to(p.dtype), conv_id, n_conv=n_conv).to(h.dtype)


def _apply_aligned(exchange: nn.Module, h: torch.Tensor, ctx: BranchContext) -> torch.Tensor:
    """Apply ``exchange`` only to each row's aligned audio slab
    ``[offset_i, offset_i + ctx.align_len)``; identity elsewhere.

    Each channel's text prefix has a different length, so the audio start
    offset differs per row; gathering each row's slab into a common
    ``(M, align_len, d)`` tensor before calling ``exchange`` aligns audio
    position ``t`` across every row of the conversation regardless of prefix
    length, exactly like the plain (unaligned) path aligns position ``t``
    when all rows share one length. CFG-style batches (see
    ``segment_conv_id``) get their offsets tiled per segment by
    ``ctx.segment_align_offsets`` - the uncond copy repeats the same packed
    layout, so its slabs sit at the identical relative offsets, not shifted.
    """
    offsets = ctx.segment_align_offsets(h.shape[0], h.device)
    length = ctx.align_len
    if int(offsets.max()) + length > h.shape[1]:
        raise RuntimeError(
            f"aligned slab [max_offset+{length}] exceeds sequence length {h.shape[1]}"
        )
    conv_id, n_conv = ctx.segment_conv_id(h.shape[0], h.device)
    idx = (offsets[:, None] + torch.arange(length, device=h.device))[..., None]
    idx = idx.expand(-1, -1, h.shape[-1])
    slab = h.gather(1, idx)
    out_slab = _call_exchange(exchange, slab, conv_id, n_conv)
    return h.scatter(1, idx, out_slab)


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

    ``copy.deepcopy`` of an injected model (e.g. for EMA) deepcopies ``ctx``
    too: the copy's blocks share one NEW context that the original's
    ``ctx.branches(...)`` does not activate. Retrieve the copy's own context
    with ``get_context(copy)``.
    """

    def __init__(self, base_block: nn.Module, exchange: nn.Module, ctx: BranchContext, spec: BlockSpec):
        super().__init__()
        self.base_block = base_block
        self.exchange = exchange
        object.__setattr__(self, "ctx", ctx)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "_fwd_snapshot", _NO_SNAPSHOT)

    def __getattr__(self, name):
        """Fall back to ``base_block`` for plain instance attributes the HF
        model reads directly off a layer object (not via ``forward``), e.g.
        Qwen3's per-layer ``attention_type`` used to pick the causal mask.
        ``nn.Module.__getattr__`` only resolves registered params/buffers/
        submodules, so unknown attributes land here; delegating makes the
        wrapper transparent to such framework introspection.

        Reaches into ``_modules`` directly (not ``self.base_block``) so that
        during ``copy.deepcopy``/unpickling of a half-built instance (no
        ``_modules`` yet), this raises the original ``AttributeError``
        instead of recursing.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            if modules is not None and "base_block" in modules:
                return getattr(modules["base_block"], name)
            raise

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
        # h.shape[1] > 1 guard: keeps single-step decode (T == 1, e.g.
        # autoregressive generation one frame at a time) on the plain path,
        # where all rows are already at the same audio step and there is no
        # per-row slab to align.
        if self.ctx.align_offsets is not None and h.shape[1] > 1:
            new_h = _apply_aligned(self.exchange, h, self.ctx)
        else:
            conv_id, n_conv = self.ctx.segment_conv_id(h.shape[0], h.device)
            new_h = _call_exchange(self.exchange, h, conv_id, n_conv)
        return self.spec.repack(out, new_h)


def _is_attr_path(target: str) -> bool:
    return all(part.isidentifier() for part in target.split("."))


def _check_no_nesting(names, target) -> None:
    """A matched block must never live inside another matched block: that is
    the silent mis-scoping failure of a loose pattern (``layers.0.self_attn``
    next to ``layers.0``)."""
    for outer in names:
        prefix = outer + "."
        for inner in names:
            if inner.startswith(prefix):
                raise ValueError(
                    f"target {target!r} matches {inner!r}, which lives inside "
                    f"another matched block {outer!r}; tighten the target so "
                    "it matches only the blocks themselves"
                )


def _near_misses(target: str, modules) -> list:
    names = [n for n in modules if n]
    for literal in sorted(re.findall(r"[A-Za-z_]{2,}", target), key=len, reverse=True):
        near = [n for n in names if literal in n]
        if near:
            return near[:5]
    return difflib.get_close_matches(target, names, n=5, cutoff=0.3)


def _match_regex(target: str, modules) -> list:
    pattern = re.compile(target)
    if pattern.groups < 1:
        raise ValueError(
            f"regex target {target!r} needs a capture group for the block "
            r'depth, e.g. r"layers\.(\d+)"'
        )
    matches = []
    for name in modules:
        m = pattern.fullmatch(name)
        if m is None:
            continue
        try:
            depth = int(m.group(1))
        except (TypeError, ValueError):
            raise ValueError(
                f"regex target {target!r} captured {m.group(1)!r} from {name!r}; "
                "the first capture group must be the block's integer depth"
            ) from None
        matches.append((name, depth))
    if not matches:
        raise ValueError(
            f"regex target {target!r} matched no module names (re.fullmatch "
            f"against named_modules()); near misses: {_near_misses(target, modules)}"
        )
    _check_no_nesting([name for name, _ in matches], target)
    by_depth: dict[int, str] = {}
    for name, depth in matches:
        if depth in by_depth:
            raise ValueError(
                f"regex target {target!r} matched both {by_depth[depth]!r} and "
                f"{name!r} at depth {depth}; models with multiple block stacks "
                "need an explicit list of module names"
            )
        by_depth[depth] = name
    depths = sorted(by_depth)
    if depths != list(range(len(depths))):
        raise ValueError(
            f"regex target {target!r} resolved depths {depths}; the capture "
            f"group must yield exactly 0..{len(depths) - 1} with no gaps"
        )
    return [by_depth[d] for d in depths]


def _lookup_names(target, modules) -> list:
    names = list(target)
    if not names:
        raise ValueError("explicit target list must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"explicit target list has duplicate names: {names!r}")
    missing = [n for n in names if n not in modules]
    if missing:
        hints = {n: _near_misses(n, modules) for n in missing}
        raise ValueError(f"target module names not found: {hints!r}")
    _check_no_nesting(names, names)
    return names


def _resolve_blocks(model: nn.Module, target) -> list:
    """Resolve ``target`` (dotted path / depth regex / explicit name list,
    see ``BlockSpec``) to an ordered list of ``(parent, key, block)`` triples
    whose position is the depth."""
    if isinstance(target, str) and _is_attr_path(target):
        obj = model
        for part in target.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, nn.ModuleList):
            raise TypeError(
                f"{target!r} on {type(model).__name__} is {type(obj).__name__}, expected nn.ModuleList"
            )
        return [(obj, str(i), block) for i, block in enumerate(obj)]
    modules = dict(model.named_modules())
    if isinstance(target, str):
        names = _match_regex(target, modules)
    else:
        names = _lookup_names(target, modules)
    triples = []
    for name in names:
        parent_name, _, key = name.rpartition(".")
        triples.append((modules[parent_name], key, modules[name]))
    return triples


def _set_block(parent: nn.Module, key: str, module: nn.Module) -> None:
    if isinstance(parent, nn.ModuleList):
        parent[int(key)] = module
    else:
        setattr(parent, key, module)


def inject_exchange(
    model: nn.Module,
    spec: BlockSpec,
    schedule: ExchangeSchedule,
    ctx: BranchContext,
) -> nn.Module:
    """Replace the block at depth ``i`` with an ``ExchangedBlock`` for every
    ``P_TAC`` depth in ``schedule``; ``P`` blocks stay untouched (original
    object, original state-dict keys). Modifies ``model`` in place and
    returns it; on error the model is left unchanged (all checks run before
    any block is replaced).
    """
    blocks = _resolve_blocks(model, spec.target)
    if schedule.depth != len(blocks):
        raise ValueError(f"schedule depth {schedule.depth} != number of blocks {len(blocks)}")
    scheduled = [i for i in range(len(blocks)) if schedule.mode(i) is Mode.P_TAC]
    already = [i for i in scheduled if isinstance(blocks[i][2], ExchangedBlock)]
    if already:
        raise ValueError(f"blocks {already} are already ExchangedBlocks; remove_exchange first")
    for i in scheduled:
        parent, key, block = blocks[i]
        _set_block(parent, key, ExchangedBlock(block, schedule.exchange_for(i), ctx, spec))
    return model


def remove_exchange(model: nn.Module, spec: BlockSpec) -> nn.Module:
    """Restore every ``ExchangedBlock`` back to its ``base_block``.

    Modifies ``model`` in place and returns it; the resulting state dict has
    exactly the original keys and values.
    """
    for parent, key, block in _resolve_blocks(model, spec.target):
        if isinstance(block, ExchangedBlock):
            _set_block(parent, key, block.base_block)
    return model


def _injected_blocks(model: nn.Module) -> list:
    """Every ``ExchangedBlock`` in the model, in module-traversal order
    (stable for a fixed architecture and schedule)."""
    return [m for m in model.modules() if isinstance(m, ExchangedBlock)]


def get_context(model: nn.Module) -> BranchContext:
    """The single ``BranchContext`` shared by the model's ``ExchangedBlock``s.

    ``copy.deepcopy`` of an injected model (e.g. for EMA) deepcopies the
    context too, so the copy's blocks share one NEW context that the
    original's ``ctx.branches(...)`` does not activate; use this helper to
    retrieve the copy's own context. Raises ``ValueError`` if the model has
    no injected blocks or its blocks hold more than one context.
    """
    contexts = {id(b.ctx): b.ctx for b in _injected_blocks(model)}
    if not contexts:
        raise ValueError("model has no ExchangedBlock; inject_exchange first")
    if len(contexts) > 1:
        raise ValueError(
            f"model's ExchangedBlocks hold {len(contexts)} different BranchContexts; expected one"
        )
    return next(iter(contexts.values()))


def exchange_state_dict(model: nn.Module) -> dict:
    """Adapter-style state dict of every injected exchange, and nothing else.

    Keys are ``"{i}.exchange.{param}"`` where ``i`` is the ``ExchangedBlock``'s
    position in module-traversal order - stable for a fixed architecture and
    schedule, and independent of wrapper nesting, so exchange checkpoints stay
    separate from backbone checkpoints (whose keys shift under wrapping).
    """
    blocks = _injected_blocks(model)
    if not blocks:
        raise ValueError("model has no ExchangedBlock; inject_exchange first")
    out = {}
    for i, block in enumerate(blocks):
        for key, value in block.exchange.state_dict().items():
            out[f"{i}.exchange.{key}"] = value
    return out


def load_exchange_state_dict(model: nn.Module, state_dict: dict, strict: bool = True) -> nn.Module:
    """Inverse of ``exchange_state_dict``; modifies ``model`` in place.

    With ``strict=True`` (default) the keys must match exactly; otherwise a
    ``RuntimeError`` lists the missing and unexpected keys.
    """
    blocks = _injected_blocks(model)
    if not blocks:
        raise ValueError("model has no ExchangedBlock; inject_exchange first")
    expected = {
        f"{i}.exchange.{key}"
        for i, block in enumerate(blocks)
        for key in block.exchange.state_dict()
    }
    missing = sorted(expected - set(state_dict))
    unexpected = sorted(set(state_dict) - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"exchange state dict mismatch; missing keys: {missing}; unexpected keys: {unexpected}"
        )
    for i, block in enumerate(blocks):
        prefix = f"{i}.exchange."
        sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        block.exchange.load_state_dict(sub, strict=strict)
    return model
