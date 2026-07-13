"""Offset-aligned exchange: each row's audio slab [offset_i, offset_i +
align_len) is exchanged, aligned across a conversation; everything else
(the per-row text prefix and any trailing content) passes through
untouched. Covers the brief's three tests plus a zero-gate bit-exactness
check on the aligned path and a CFG-segment independence check."""

import pytest
import torch

from ..exchange import IdentityExchange, TACExchange
from ..inject import BranchContext, _apply_aligned


def _ctx(counts, offsets, length):
    ctx = BranchContext()
    mgr = ctx.branches(counts=counts, align_offsets=offsets, align_len=length)
    return ctx, mgr


def test_prefix_untouched_gate_open():
    torch.manual_seed(0)
    ex = TACExchange(8)
    torch.nn.init.ones_(ex.g)
    h = torch.randn(2, 12, 8)
    offsets = torch.tensor([3, 5])  # row 0 audio at [3, 9), row 1 at [5, 11)
    ctx, mgr = _ctx([2], offsets, 6)
    with mgr:
        out = _apply_aligned(ex, h, ctx)
    assert torch.equal(out[0, :3], h[0, :3]) and torch.equal(out[0, 9:], h[0, 9:])
    assert torch.equal(out[1, :5], h[1, :5]) and torch.equal(out[1, 11:], h[1, 11:])
    assert not torch.allclose(out[0, 3:9], h[0, 3:9])


def test_aligned_equals_manual_roll():
    torch.manual_seed(0)
    ex = TACExchange(8)
    torch.nn.init.ones_(ex.g)
    h = torch.randn(2, 12, 8)
    offsets = torch.tensor([3, 5])
    ctx, mgr = _ctx([2], offsets, 6)
    with mgr:
        out = _apply_aligned(ex, h, ctx)
    # manual reference: gather each row's audio slab, run plain exchange, compare
    slab = torch.stack([h[0, 3:9], h[1, 5:11]])
    ref = ex(slab, torch.tensor([0, 0]), n_conv=1)
    assert torch.allclose(out[0, 3:9], ref[0], atol=1e-6)
    assert torch.allclose(out[1, 5:11], ref[1], atol=1e-6)


def test_offset_validation():
    ctx = BranchContext()
    with pytest.raises(ValueError):
        with ctx.branches(counts=[2], align_offsets=torch.tensor([0]), align_len=4):
            pass  # offsets length != sum(counts)
    with pytest.raises(ValueError):
        with ctx.branches(counts=[2], align_offsets=torch.tensor([0, 1]), align_len=0):
            pass  # non-positive length


def test_apply_aligned_rejects_out_of_bounds_slab():
    """``_apply_aligned`` itself must guard the slab bound: an offset whose
    slab would run past the sequence length raises RuntimeError, distinct
    from the ValueError that ``branches(...)`` raises for malformed
    offsets/length up front."""
    h = torch.randn(2, 12, 8)
    ex = TACExchange(8)
    offsets = torch.tensor([8, 8])  # slab [8, 14) but T is only 12
    ctx, mgr = _ctx([2], offsets, 6)
    with mgr, pytest.raises(RuntimeError, match="exceeds sequence length"):
        _apply_aligned(ex, h, ctx)


# ---- zero-gate bit-exactness on the aligned path ----


def test_zero_gate_identity_with_offsets_active():
    """The identity-at-init guarantee must hold on the aligned path too: a
    zero-gated TACExchange leaves the whole tensor (prefix AND audio slab)
    bit-exactly unchanged, exactly like the plain (non-aligned) path."""
    torch.manual_seed(1)
    ex = TACExchange(8)  # g stays at its zero init
    h = torch.randn(2, 12, 8)
    offsets = torch.tensor([3, 5])
    ctx, mgr = _ctx([2], offsets, 6)
    with mgr:
        out = _apply_aligned(ex, h, ctx)
    assert torch.equal(out, h)


# ---- CFG-segment independence ----


def test_cfg_segment_offsets_independent():
    """A CFG-style batch concatenates a second (uncond) copy of the packed
    rows along the batch axis; segment_align_offsets must tile the base
    offsets per segment so each segment's slab aligns to the SAME relative
    offsets, and the two segments exchange fully independently (no mixing
    across the batch concat, mirroring segment_conv_id's CFG tests)."""
    torch.manual_seed(2)
    ex = TACExchange(8)
    torch.nn.init.ones_(ex.g)
    base = torch.randn(2, 12, 8)
    uncond = torch.randn(2, 12, 8)
    h = torch.cat((base, uncond))  # (4, 12, 8): rows 0-1 cond, rows 2-3 uncond
    offsets = torch.tensor([3, 5])
    ctx, mgr = _ctx([2], offsets, 6)
    with mgr:
        out_cat = _apply_aligned(ex, h, ctx)
        out_base_only = _apply_aligned(ex, base, ctx)
        out_uncond_only = _apply_aligned(ex, uncond, ctx)

    assert torch.allclose(out_cat[:2], out_base_only, atol=1e-6)
    assert torch.allclose(out_cat[2:], out_uncond_only, atol=1e-6)
    # untouched regions still pass through for both segments (row 2 mirrors
    # row 0's offset 3, row 3 mirrors row 1's offset 5 - tiled, not repeated)
    assert torch.equal(out_cat[0, :3], h[0, :3]) and torch.equal(out_cat[0, 9:], h[0, 9:])
    assert torch.equal(out_cat[2, :3], h[2, :3]) and torch.equal(out_cat[2, 9:], h[2, 9:])
    assert torch.equal(out_cat[1, :5], h[1, :5]) and torch.equal(out_cat[1, 11:], h[1, 11:])
    assert torch.equal(out_cat[3, :5], h[3, :5]) and torch.equal(out_cat[3, 11:], h[3, 11:])
