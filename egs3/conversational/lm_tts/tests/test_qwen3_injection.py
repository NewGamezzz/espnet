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
