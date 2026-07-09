"""Injection tests against a small random-init espnet2 F5 DiT: zero-init
identity vs the uninjected model, conversation isolation, count
generalization, gradient flow, injection guards (identity / inactive ctx /
state-dict restore / CFG segments / activation checkpointing), schedule
parsing, and import purity."""

import copy
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
    exchange_state_dict,
    get_context,
    inject_exchange,
    load_exchange_state_dict,
    remove_exchange,
)
from conftest import (
    DEPTH,
    DIM,
    MEL,
    SRC_DIR,
    T,
    inject_all,
    iter_exchanges,
    make_dit,
    make_packed_inputs,
    randomize_exchanges,
    set_gates,
    slice_conversation,
)

FACTORIES = {
    "tac": lambda: TACExchange(DIM),
    "mha": lambda: BranchMHAExchange(DIM, n_heads=4),
}

COUNTS = (2, 3)


def cfg_cat(inputs):
    """CFG-style batch: the packed inputs with an uncond copy concatenated."""
    return tuple(torch.cat((t_, torch.zeros_like(t_))) for t_ in inputs)


# ---- zero-init identity ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("ckpt", [False, True])
def test_zero_init_identity(kind, ckpt):
    model = make_dit(checkpoint_activations=ckpt)
    inputs = make_packed_inputs(COUNTS)
    with torch.no_grad():
        ref = model(*inputs)
    ctx = inject_all(model, FACTORIES[kind])
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out = model(*inputs)
    assert torch.equal(out, ref)


# ---- conversation isolation + count generalization ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_conversation_isolation(kind):
    """A ragged packed batch matches each conversation run on its own through
    the whole injected backbone, so conversations never mix and no padding is
    ever needed."""
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
            with ctx.branches(counts=(n,)):
                alone = model(*slice_conversation(inputs, COUNTS, i))
            assert torch.allclose(out[start : start + n], alone, atol=1e-5)
            start += n


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_count_generalization(kind):
    """One injected model serves any branch-count mix without reconfiguration."""
    model = make_dit()
    ctx = inject_all(model, FACTORIES[kind])
    set_gates(model, 0.5)
    randomize_exchanges(model)
    for counts in ((2,), (2, 2), (3, 1), (4, 2, 3)):
        inputs = make_packed_inputs(counts, seed=sum(counts))
        with torch.no_grad(), ctx.branches(counts=counts):
            out = model(*inputs)
        assert out.shape == (sum(counts), T, MEL)


# ---- gradient flow ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("ckpt", [False, True])
def test_gradient_flow(kind, ckpt):
    model = make_dit(checkpoint_activations=ckpt)
    ctx = inject_all(model, FACTORIES[kind])
    inputs = make_packed_inputs(COUNTS)

    with ctx.branches(counts=COUNTS):
        model(*inputs).sum().backward()
    gates = [m.g for m in iter_exchanges(model)]
    assert len(gates) == DEPTH
    for g in gates:
        assert g.grad is not None and g.grad.abs().item() > 0

    model.zero_grad(set_to_none=True)
    set_gates(model, 0.5)
    with ctx.branches(counts=COUNTS):
        model(*inputs).sum().backward()
    for m in iter_exchanges(model):
        for name, p in m.named_parameters():
            assert p.grad is not None, name
            assert p.grad.abs().sum().item() > 0, name


# ---- checkpoint recompute guard: backward must stay inside ctx.branches ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_checkpoint_backward_outside_context_fails(kind):
    model = make_dit(checkpoint_activations=True)
    ctx = inject_all(model, FACTORIES[kind])
    inputs = make_packed_inputs(COUNTS)

    with ctx.branches(counts=COUNTS):
        loss = model(*inputs).sum()
    with pytest.raises(RuntimeError, match="BranchContext changed"):
        loss.backward()

    # Re-entering the context (even with the same counts) is a new context:
    # the recompute must see the very same activation, not a lookalike.
    model.zero_grad(set_to_none=True)
    with ctx.branches(counts=COUNTS):
        loss = model(*inputs).sum()
    with ctx.branches(counts=COUNTS), pytest.raises(RuntimeError, match="BranchContext changed"):
        loss.backward()


def test_backward_outside_context_ok_without_checkpointing():
    model = make_dit(checkpoint_activations=False)
    ctx = inject_all(model, FACTORIES["tac"])
    inputs = make_packed_inputs(COUNTS)

    with ctx.branches(counts=COUNTS):
        loss = model(*inputs).sum()
    loss.backward()
    for m in iter_exchanges(model):
        assert m.g.grad is not None


# ---- identity exchange / inactive ctx guards ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_injection_guard(kind):
    inputs = make_packed_inputs(COUNTS)
    with torch.no_grad():
        ref = make_dit()(*inputs)

    model = make_dit()
    ctx = inject_all(model, IdentityExchange)
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out_identity = model(*inputs)
    assert torch.equal(out_identity, ref)

    model = make_dit()
    inject_all(model, FACTORIES[kind])  # ctx stays inactive
    with torch.no_grad():
        out_inactive = model(*inputs)
    assert torch.equal(out_inactive, ref)


# ---- deepcopy (EMA) detaches the context: get_context recovers it ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_get_context_after_deepcopy(kind):
    model = make_dit()
    ctx = inject_all(model, FACTORIES[kind])
    set_gates(model, 0.5)
    randomize_exchanges(model)
    inputs = make_packed_inputs(COUNTS, seed=23)
    model_copy = copy.deepcopy(model)

    with torch.no_grad():
        base = model(*inputs)  # ctx inactive: exchanges skipped
        with ctx.branches(counts=COUNTS):
            out_orig = model(*inputs)
            # The original's context does NOT activate the deepcopy: its
            # blocks share a deepcopied context of their own.
            out_copy_inactive = model_copy(*inputs)
        copy_ctx = get_context(model_copy)
        assert copy_ctx is not ctx
        with copy_ctx.branches(counts=COUNTS):
            out_copy = model_copy(*inputs)

    assert torch.equal(out_copy_inactive, base)
    assert not torch.equal(out_orig, base)
    assert torch.equal(out_copy, out_orig)


def test_get_context_validation():
    with pytest.raises(ValueError, match="no ExchangedBlock"):
        get_context(make_dit())

    model = make_dit()
    sched_a = ExchangeSchedule.from_spec({"1-2": "P+TAC", "3-4": "P"}, depth=DEPTH, factory=FACTORIES["tac"])
    sched_b = ExchangeSchedule.from_spec({"1-2": "P", "3-4": "P+TAC"}, depth=DEPTH, factory=FACTORIES["tac"])
    inject_exchange(model, REGISTRY["f5_dit"], sched_a, BranchContext())
    inject_exchange(model, REGISTRY["f5_dit"], sched_b, BranchContext())
    with pytest.raises(ValueError, match="different BranchContexts"):
        get_context(model)


# ---- adapter-style exchange-only checkpoints ----


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_exchange_state_dict_roundtrip(kind):
    model_a = make_dit()
    inject_all(model_a, FACTORIES[kind])
    set_gates(model_a, 0.5)
    randomize_exchanges(model_a, seed=7)
    sd = exchange_state_dict(model_a)
    assert sd and all(".exchange." in key for key in sd)

    model_b = make_dit(seed=1)
    inject_all(model_b, FACTORIES[kind])
    load_exchange_state_dict(model_b, sd)
    for ex_a, ex_b in zip(iter_exchanges(model_a), iter_exchanges(model_b)):
        for (name_a, p_a), (name_b, p_b) in zip(ex_a.named_parameters(), ex_b.named_parameters()):
            assert name_a == name_b
            assert torch.equal(p_a, p_b), name_a

    with pytest.raises(RuntimeError, match="missing"):
        load_exchange_state_dict(model_b, dict(list(sd.items())[1:]))
    with pytest.raises(RuntimeError, match="unexpected"):
        load_exchange_state_dict(model_b, {**sd, "9.exchange.bogus": torch.zeros(())})


def test_exchange_state_dict_requires_injection():
    with pytest.raises(ValueError, match="no ExchangedBlock"):
        exchange_state_dict(make_dit())
    with pytest.raises(ValueError, match="no ExchangedBlock"):
        load_exchange_state_dict(make_dit(), {})


# ---- remove_exchange restores the exact state dict ----


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


# ---- CFG-style segment safety ----


def test_cfg_segments_identity():
    model = make_dit()
    cat_inputs = cfg_cat(make_packed_inputs(COUNTS))
    with torch.no_grad():
        ref = model(*cat_inputs)

    ctx = inject_all(model, IdentityExchange)
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out = model(*cat_inputs)
    assert torch.equal(out, ref)


def test_cfg_segments_no_cross_mixing():
    model = make_dit()
    ctx = inject_all(model, FACTORIES["tac"])
    set_gates(model, 0.5)
    randomize_exchanges(model)

    inputs = make_packed_inputs(COUNTS, seed=17)
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out_first_only = model(*inputs)
        out_cat = model(*cfg_cat(inputs))
    assert torch.allclose(out_cat[: inputs[0].shape[0]], out_first_only, atol=1e-5)


def test_row_count_must_be_multiple_of_total():
    model = make_dit()
    ctx = inject_all(model, FACTORIES["tac"])
    inputs = make_packed_inputs((2, 2))  # 4 rows, but the context declares 5
    with ctx.branches(counts=COUNTS), pytest.raises(RuntimeError, match="not a multiple"):
        model(*inputs)


def test_branches_argument_validation():
    ctx = BranchContext()
    with pytest.raises(ValueError):
        with ctx.branches(counts=()):
            pass
    with pytest.raises(ValueError):
        with ctx.branches(counts=(2, 0)):
            pass
    assert not ctx.active


def test_branches_is_not_reentrant():
    ctx = BranchContext()
    with ctx.branches(counts=(2,)):
        with pytest.raises(RuntimeError, match="already active"):
            with ctx.branches(counts=(3,)):
                pass
        assert ctx.active  # outer state survives the rejected inner activation
    assert not ctx.active


def test_failed_injection_is_atomic():
    model = make_dit()
    orig_blocks = list(model.transformer_blocks)
    partial = ExchangeSchedule.from_spec(
        {"1": "P", "2": "P+TAC", "3-4": "P"}, depth=DEPTH, factory=FACTORIES["tac"]
    )
    inject_exchange(model, REGISTRY["f5_dit"], partial, BranchContext())
    wrapped = model.transformer_blocks[1]

    full = ExchangeSchedule.from_spec({f"1-{DEPTH}": "P+TAC"}, depth=DEPTH, factory=FACTORIES["tac"])
    with pytest.raises(ValueError, match="already ExchangedBlocks"):
        inject_exchange(model, REGISTRY["f5_dit"], full, BranchContext())
    # Nothing was mutated: block 0 (scheduled BEFORE the offending index) is
    # still the original, not a fresh ExchangedBlock.
    assert model.transformer_blocks[0] is orig_blocks[0]
    assert model.transformer_blocks[1] is wrapped
    assert model.transformer_blocks[2] is orig_blocks[2]
    assert model.transformer_blocks[3] is orig_blocks[3]


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


# ---- import purity ----


def test_import_purity():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import branch_exchange\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('espnet2', 'espnet3'))\n"
        "assert not bad, f'branch_exchange imported espnet modules: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
