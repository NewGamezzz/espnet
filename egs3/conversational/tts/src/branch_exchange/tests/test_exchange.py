"""Module-level tests for the exchange contract: (B, N, T, d) -> (B, N, T, d),
zero-init identity, permutation equivariance, branch-count generalization,
and pad-mask semantics. No espnet imports here."""

import pytest
import torch

from branch_exchange import BranchMHAExchange, IdentityExchange, TACExchange

DIM = 32
B, T = 2, 5


def make_exchange(kind, dim=DIM):
    if kind == "tac":
        return TACExchange(dim)
    if kind == "mha":
        return BranchMHAExchange(dim, n_heads=4)
    raise ValueError(kind)


def randomize(module, seed=0, gate=0.5):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name == "g":
                p.fill_(gate)
            else:
                p.copy_(0.5 * torch.randn(p.shape, generator=gen))
    return module


def make_h(n_branch, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, n_branch, T, DIM, generator=gen)


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("with_pad", [False, True])
def test_zero_init_identity(kind, with_pad):
    ex = make_exchange(kind)
    h = make_h(3)
    pad = torch.tensor([[False, False, True]] * B) if with_pad else None
    with torch.no_grad():
        out = ex(h, pad_mask=pad)
    assert torch.equal(out, h)


@pytest.mark.parametrize("kind", ["tac", "mha"])
@pytest.mark.parametrize("with_pad", [False, True])
def test_permutation_equivariance(kind, with_pad):
    n = 4
    ex = randomize(make_exchange(kind))
    h = make_h(n)
    pad = torch.tensor([[False, False, True, False], [False, True, False, False]]) if with_pad else None
    torch.manual_seed(1)
    perm = torch.randperm(n)
    with torch.no_grad():
        out = ex(h, pad_mask=pad)
        out_perm = ex(h[:, perm], pad_mask=pad[:, perm] if pad is not None else None)
    assert torch.allclose(out_perm, out[:, perm], atol=1e-5)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_any_branch_count(kind):
    ex = randomize(make_exchange(kind))
    for n in (1, 2, 3, 4):
        h = make_h(n, seed=n)
        with torch.no_grad():
            out = ex(h)
        assert out.shape == h.shape


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_padding_equals_absence(kind):
    """A fully padded ghost branch must not influence the real branches."""
    ex = randomize(make_exchange(kind))
    h2 = make_h(2, seed=7)
    ghost = torch.randn(B, 1, T, DIM, generator=torch.Generator().manual_seed(99))
    h3 = torch.cat((h2, ghost), dim=1)
    pad = torch.tensor([[False, False, True]] * B)
    with torch.no_grad():
        out2 = ex(h2)
        out3 = ex(h3, pad_mask=pad)
    assert torch.allclose(out3[:, :2], out2, atol=1e-5)


def test_identity_exchange():
    h = make_h(3)
    ex = IdentityExchange()
    out = ex(h, pad_mask=torch.zeros(B, 3, dtype=torch.bool))
    assert torch.equal(out, h)
