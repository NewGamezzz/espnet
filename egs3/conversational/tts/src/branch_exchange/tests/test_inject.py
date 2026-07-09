"""Injection tests against a small random-init espnet2 F5 DiT: zero-init
identity vs independent passes, count generalization, gradient flow,
injection guards (identity / inactive ctx / state-dict restore / CFG
segments / activation checkpointing), schedule parsing, and import purity."""

import subprocess
import sys

import pytest
import torch

from branch_exchange import (
    REGISTRY,
    BranchContext,
    BranchMHAExchange,
    ExchangedBlock,
    ExchangeSchedule,
    IdentityExchange,
    Mode,
    TACExchange,
    inject_exchange,
    remove_exchange,
)
from conftest import (
    B,
    DEPTH,
    DIM,
    MEL,
    SRC_DIR,
    T,
    inject_all,
    iter_exchanges,
    make_branch_inputs,
    make_dit,
    make_packed_inputs,
    randomize_exchanges,
    run_folded,
    run_independent,
    set_gates,
    slice_conversation,
)

FACTORIES = {
    "tac": lambda: TACExchange(DIM),
    "mha": lambda: BranchMHAExchange(DIM, n_heads=4),
}


# ---- test 1 (and 5d forward half): zero-init identity ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("ckpt", [False, True])
def test_zero_init_identity(kind, ckpt):
    # The brief asks for torch.equal against independent per-branch passes, but
    # on this platform even the UNINJECTED DiT differs by ~1 ULP (max 6e-8)
    # between a batch-6 and a batch-2 run (batch-size-dependent BLAS kernels).
    # So the bit-equality claim "g=0 exchanges are exactly the identity" is
    # asserted against the uninjected model on the same folded batch, and the
    # fold/unfold correctness against independent passes uses allclose.
    model = make_dit(checkpoint_activations=ckpt)
    inputs = make_branch_inputs(3)
    with torch.no_grad():
        ref_folded = run_folded(model, inputs)
        ref_independent = run_independent(model, inputs)
    ctx = inject_all(model, FACTORIES[kind])
    with torch.no_grad(), ctx.branches(3):
        out = run_folded(model, inputs)
    assert torch.equal(out, ref_folded)
    assert torch.allclose(out, ref_independent, atol=1e-6)


# ---- test 3: count generalization + padding == absence ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_count_generalization(kind):
    model = make_dit()
    ctx = inject_all(model, FACTORIES[kind])
    set_gates(model, 0.5)
    randomize_exchanges(model)

    for n in (2, 3, 4):
        inputs = make_branch_inputs(n, seed=n)
        with torch.no_grad(), ctx.branches(n):
            out = run_folded(model, inputs)
        assert out.shape == (B, n, T, MEL)

    inputs2 = make_branch_inputs(2, seed=7)
    ghost = make_branch_inputs(1, seed=99)
    inputs3 = tuple(torch.cat((real, g_), dim=1) for real, g_ in zip(inputs2, ghost))
    pad = torch.tensor([[False, False, True]] * B)
    with torch.no_grad():
        with ctx.branches(2):
            out2 = run_folded(model, inputs2)
        with ctx.branches(3, pad_mask=pad):
            out3 = run_folded(model, inputs3)
    assert torch.allclose(out3[:, :2], out2, atol=1e-5)


# ---- test 4 (and 5d backward half): gradient flow ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("ckpt", [False, True])
def test_gradient_flow(kind, ckpt):
    model = make_dit(checkpoint_activations=ckpt)
    ctx = inject_all(model, FACTORIES[kind])
    inputs = make_branch_inputs(3)

    with ctx.branches(3):
        run_folded(model, inputs).sum().backward()
    gates = [m.g for m in iter_exchanges(model)]
    assert len(gates) == DEPTH
    for g in gates:
        assert g.grad is not None and g.grad.abs().item() > 0

    model.zero_grad(set_to_none=True)
    set_gates(model, 0.5)
    with ctx.branches(3):
        run_folded(model, inputs).sum().backward()
    for m in iter_exchanges(model):
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert p.grad.abs().sum().item() > 0, name


# ---- checkpoint recompute guard: backward must stay inside ctx.branches ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_checkpoint_backward_outside_context_fails(kind):
    model = make_dit(checkpoint_activations=True)
    ctx = inject_all(model, FACTORIES[kind])
    inputs = make_branch_inputs(3)

    with ctx.branches(3):
        loss = run_folded(model, inputs).sum()
    with pytest.raises(RuntimeError, match="BranchContext changed"):
        loss.backward()

    model.zero_grad(set_to_none=True)
    with ctx.branches(3):
        loss = run_folded(model, inputs).sum()
    with ctx.branches(2), pytest.raises(RuntimeError, match="BranchContext changed"):
        loss.backward()


def test_backward_outside_context_ok_without_checkpointing():
    model = make_dit(checkpoint_activations=False)
    ctx = inject_all(model, FACTORIES["tac"])
    inputs = make_branch_inputs(3)

    with ctx.branches(3):
        loss = run_folded(model, inputs).sum()
    loss.backward()
    for m in iter_exchanges(model):
        assert m.g.grad is not None


# ---- test 5a: identity exchange / inactive ctx guards ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_injection_guard(kind):
    inputs = make_branch_inputs(3)
    with torch.no_grad():
        ref = run_folded(make_dit(), inputs)

    model = make_dit()
    ctx = inject_all(model, IdentityExchange)
    with torch.no_grad(), ctx.branches(3):
        out_identity = run_folded(model, inputs)
    assert torch.equal(out_identity, ref)

    model = make_dit()
    inject_all(model, FACTORIES[kind])  # ctx stays inactive
    with torch.no_grad():
        out_inactive = run_folded(model, inputs)
    assert torch.equal(out_inactive, ref)


# ---- test 5b: remove_exchange restores the exact state dict ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_remove_restores_state_dict(kind):
    model = make_dit()
    orig_sd = {k: v.clone() for k, v in model.state_dict().items()}
    inject_all(model, FACTORIES[kind])
    assert set(model.state_dict()) != set(orig_sd)

    remove_exchange(model, REGISTRY["f5_dit"])
    sd = model.state_dict()
    assert set(sd) == set(orig_sd)
    for key, value in orig_sd.items():
        assert torch.equal(sd[key], value), key


# ---- test 5c: CFG-style segment safety ----


def test_cfg_segments_identity():
    model = make_dit()
    x, cond, text, time = (t_.flatten(0, 1) for t_ in make_branch_inputs(3))
    cat_inputs = (
        torch.cat((x, torch.zeros_like(x))),
        torch.cat((cond, torch.zeros_like(cond))),
        torch.cat((text, torch.zeros_like(text))),
        torch.cat((time, torch.zeros_like(time))),
    )
    with torch.no_grad():
        ref = model(*cat_inputs)

    ctx = inject_all(model, IdentityExchange)
    with torch.no_grad(), ctx.branches(3):
        out = model(*cat_inputs)
    assert torch.equal(out, ref)


def test_cfg_segments_no_cross_mixing():
    model = make_dit()
    ctx = inject_all(model, FACTORIES["tac"])
    set_gates(model, 0.5)
    randomize_exchanges(model)

    x, cond, text, time = (t_.flatten(0, 1) for t_ in make_branch_inputs(3))
    with torch.no_grad(), ctx.branches(3):
        out_first_only = model(x, cond, text, time)
        out_cat = model(
            torch.cat((x, torch.zeros_like(x))),
            torch.cat((cond, torch.zeros_like(cond))),
            torch.cat((text, torch.zeros_like(text))),
            torch.cat((time, torch.zeros_like(time))),
        )
    assert torch.allclose(out_cat[: x.shape[0]], out_first_only, atol=1e-5)


# ---- packed (ragged, padding-free) mode through the injected backbone ----

COUNTS = (2, 3)


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("ckpt", [False, True])
def test_packed_zero_init_identity(kind, ckpt):
    model = make_dit(checkpoint_activations=ckpt)
    inputs = make_packed_inputs(COUNTS)
    with torch.no_grad():
        ref = model(*inputs)
    ctx = inject_all(model, FACTORIES[kind])
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out = model(*inputs)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_packed_equals_per_conversation(kind):
    """A ragged packed batch matches each conversation run on its own through
    the rectangular path, so conversations never mix and no padding is needed."""
    model = make_dit()
    ctx = inject_all(model, FACTORIES[kind])
    set_gates(model, 0.5)
    randomize_exchanges(model)
    inputs = make_packed_inputs(COUNTS, seed=13)
    with torch.no_grad():
        with ctx.branches(counts=COUNTS):
            out = model(*inputs)
        start = 0
        for i, n in enumerate(COUNTS):
            with ctx.branches(n):
                ref = run_folded(model, slice_conversation(inputs, COUNTS, i))
            assert torch.allclose(out[start : start + n], ref[0], atol=1e-5)
            start += n


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_packed_matches_rectangular(kind):
    """With uniform counts, packed mode agrees with the rectangular n= path."""
    model = make_dit()
    ctx = inject_all(model, FACTORIES[kind])
    set_gates(model, 0.5)
    randomize_exchanges(model)
    inputs = make_branch_inputs(3)  # (B=2, N=3, ...)
    flat = tuple(t_.flatten(0, 1) for t_ in inputs)
    with torch.no_grad():
        with ctx.branches(3):
            ref = run_folded(model, inputs)
        with ctx.branches(counts=(3, 3)):
            out = model(*flat)
    assert torch.allclose(out.unflatten(0, (B, 3)), ref, atol=1e-5)


def test_packed_cfg_segments_no_cross_mixing():
    model = make_dit()
    ctx = inject_all(model, FACTORIES["tac"])
    set_gates(model, 0.5)
    randomize_exchanges(model)
    x, cond, text, time = make_packed_inputs(COUNTS, seed=17)
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out_first_only = model(x, cond, text, time)
        out_cat = model(
            torch.cat((x, torch.zeros_like(x))),
            torch.cat((cond, torch.zeros_like(cond))),
            torch.cat((text, torch.zeros_like(text))),
            torch.cat((time, torch.zeros_like(time))),
        )
    assert torch.allclose(out_cat[: x.shape[0]], out_first_only, atol=1e-5)


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("ckpt", [False, True])
def test_packed_gradient_flow(kind, ckpt):
    model = make_dit(checkpoint_activations=ckpt)
    ctx = inject_all(model, FACTORIES[kind])
    set_gates(model, 0.5)
    inputs = make_packed_inputs(COUNTS)
    with ctx.branches(counts=COUNTS):
        model(*inputs).sum().backward()
    for m in iter_exchanges(model):
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert p.grad.abs().sum().item() > 0, name


def test_packed_checkpoint_backward_outside_context_fails():
    model = make_dit(checkpoint_activations=True)
    ctx = inject_all(model, FACTORIES["tac"])
    inputs = make_packed_inputs(COUNTS)
    with ctx.branches(counts=COUNTS):
        loss = model(*inputs).sum()
    with pytest.raises(RuntimeError, match="BranchContext changed"):
        loss.backward()


def test_packed_row_count_must_be_multiple_of_total():
    model = make_dit()
    ctx = inject_all(model, FACTORIES["tac"])
    inputs = make_packed_inputs((2, 2))  # 4 rows, but the context declares 5
    with ctx.branches(counts=COUNTS), pytest.raises(RuntimeError, match="not a multiple"):
        model(*inputs)


def test_branches_argument_validation():
    ctx = BranchContext()
    with pytest.raises(ValueError):
        with ctx.branches():
            pass
    with pytest.raises(ValueError):
        with ctx.branches(3, counts=COUNTS):
            pass
    with pytest.raises(ValueError):
        with ctx.branches(counts=()):
            pass
    with pytest.raises(ValueError):
        with ctx.branches(counts=(2, 0)):
            pass
    with pytest.raises(ValueError):
        with ctx.branches(counts=COUNTS, pad_mask=torch.zeros(1, 5, dtype=torch.bool)):
            pass
    assert not ctx.active and not ctx.packed


# ---- schedule parsing / partial injection ----


def test_schedule_spec_parsing_and_validation():
    factory = IdentityExchange
    sched = ExchangeSchedule.from_spec({"1-2": "P", "3": "P+TAC", "4-4": "P_TAC"}, depth=4, factory=factory)
    assert [sched.mode(i) for i in range(4)] == [Mode.P, Mode.P, Mode.P_TAC, Mode.P_TAC]
    assert sched.exchange_for(0) is None
    assert isinstance(sched.exchange_for(2), IdentityExchange)
    assert sched.exchange_for(2) is not sched.exchange_for(3)

    with pytest.raises(ValueError):
        ExchangeSchedule.from_spec({"1-2": "P"}, depth=4, factory=factory)
    with pytest.raises(ValueError):
        ExchangeSchedule.from_spec({"1-3": "P", "3-4": "P"}, depth=4, factory=factory)
    with pytest.raises(ValueError):
        ExchangeSchedule.from_spec({"1-4": "P", "5": "P"}, depth=4, factory=factory)
    with pytest.raises(NotImplementedError):
        ExchangeSchedule.from_spec({"1-4": "M"}, depth=4, factory=factory)
    with pytest.raises(ValueError):
        ExchangeSchedule.from_spec({"1-4": "X"}, depth=4, factory=factory)


def test_p_blocks_stay_untouched():
    model = make_dit()
    orig_blocks = list(model.transformer_blocks)
    sched = ExchangeSchedule.from_spec({"1-2": "P", "3-4": "P+TAC"}, depth=4, factory=FACTORIES["tac"])
    inject_exchange(model, REGISTRY["f5_dit"], sched, BranchContext())

    assert model.transformer_blocks[0] is orig_blocks[0]
    assert model.transformer_blocks[1] is orig_blocks[1]
    assert isinstance(model.transformer_blocks[2], ExchangedBlock)
    assert model.transformer_blocks[2].base_block is orig_blocks[2]
    assert isinstance(model.transformer_blocks[3], ExchangedBlock)


# ---- test 6: import purity ----


def test_import_purity():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import branch_exchange\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('espnet2', 'espnet3'))\n"
        "assert not bad, f'branch_exchange imported espnet modules: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
