import torch

from src.branch_exchange import (
    BranchContext, ExchangeSchedule, TACExchange,
    inject_exchange, remove_exchange,
)
from src.branch_exchange.registry import REGISTRY


def _inject(model, dim=64, spec_ranges={"1-2": "P", "3-4": "P+TAC"}):
    ctx = BranchContext()
    schedule = ExchangeSchedule.from_spec(spec_ranges, depth=4, factory=lambda: TACExchange(dim))
    inject_exchange(model, REGISTRY["qwen3"], schedule, ctx)
    return ctx


def test_zero_init_identity(tiny_qwen3):
    ids = torch.randint(0, 128, (4, 16))
    base = tiny_qwen3(ids).logits.clone()
    ctx = _inject(tiny_qwen3)
    with torch.no_grad():
        inactive = tiny_qwen3(ids).logits
        with ctx.branches(counts=[2, 2]):
            active_zero_gate = tiny_qwen3(ids).logits
    assert torch.equal(inactive, base)
    assert torch.equal(active_zero_gate, base)


def test_remove_restores_state_dict(tiny_qwen3):
    before = {k: v.clone() for k, v in tiny_qwen3.state_dict().items()}
    _inject(tiny_qwen3)
    remove_exchange(tiny_qwen3, REGISTRY["qwen3"])
    after = tiny_qwen3.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_permutation_equivariance(tiny_qwen3):
    ctx = _inject(tiny_qwen3)
    for blk in [m for m in tiny_qwen3.modules() if type(m).__name__ == "ExchangedBlock"]:
        torch.nn.init.ones_(blk.exchange.g)  # open the gate
    ids = torch.randint(0, 128, (3, 16))
    perm = torch.tensor([1, 2, 0])  # permute branches within the single conversation
    with torch.no_grad(), ctx.branches(counts=[3]):
        out = tiny_qwen3(ids).logits
        out_perm = tiny_qwen3(ids[perm]).logits
    assert torch.allclose(out[perm], out_perm, atol=1e-5)


def test_zero_init_identity_bf16_backbone(tiny_qwen3):
    """A bf16 backbone with fp32 exchanges (the dtype `factory` produces) must
    run and stay bit-exact at zero gate - the exact mixed-dtype configuration
    that crashed the real-checkpoint parity run before `_call_exchange`
    adapted activations to the exchange dtype (Delta gate, 2026-07-14)."""
    model = tiny_qwen3.to(torch.bfloat16)
    ids = torch.randint(0, 128, (4, 16))
    with torch.no_grad():
        base = model(ids).logits.clone()
    ctx = _inject(model)
    exchange_params = [
        p for m in model.modules() if type(m).__name__ == "ExchangedBlock"
        for p in m.exchange.parameters()
    ]
    assert exchange_params and all(p.dtype == torch.float32 for p in exchange_params)
    with torch.no_grad():
        with ctx.branches(counts=[2, 2]):
            plain = model(ids).logits
        with ctx.branches(counts=[2, 2], align_offsets=torch.tensor([3, 5, 2, 4]), align_len=8):
            aligned = model(ids).logits
    assert plain.dtype == torch.bfloat16 and aligned.dtype == torch.bfloat16
    assert torch.equal(plain, base)
    assert torch.equal(aligned, base)


def test_gradient_flow_bf16_backbone(tiny_qwen3):
    """Gradients must reach the fp32 exchange params through the bf16
    backbone (the activation casts sit on the autograd path)."""
    model = tiny_qwen3.to(torch.bfloat16)
    ids = torch.randint(0, 128, (2, 16))
    ctx = _inject(model)
    with ctx.branches(counts=[2]):
        loss = model(ids).logits.float().sum()
    loss.backward()
    gates = [
        m.exchange.g for m in model.modules() if type(m).__name__ == "ExchangedBlock"
    ]
    assert gates
    for g in gates:
        assert g.grad is not None and torch.isfinite(g.grad).all()
