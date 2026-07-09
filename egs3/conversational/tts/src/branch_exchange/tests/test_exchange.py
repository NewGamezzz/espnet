"""Module-level tests for the exchange contract (dense and packed layouts):
zero-init identity, permutation equivariance, branch-count generalization,
and conversation isolation. No espnet imports here."""

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
def test_zero_init_identity(kind):
    ex = make_exchange(kind)
    h = make_h(3)
    with torch.no_grad():
        out = ex(h)
    assert torch.equal(out, h)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_permutation_equivariance(kind):
    n = 4
    ex = randomize(make_exchange(kind))
    h = make_h(n)
    torch.manual_seed(1)
    perm = torch.randperm(n)
    with torch.no_grad():
        out = ex(h)
        out_perm = ex(h[:, perm])
    assert torch.allclose(out_perm, out[:, perm], atol=1e-5)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_any_branch_count(kind):
    ex = randomize(make_exchange(kind))
    for n in (1, 2, 3, 4):
        h = make_h(n, seed=n)
        with torch.no_grad():
            out = ex(h)
        assert out.shape == h.shape


def test_identity_exchange():
    h = make_h(3)
    ex = IdentityExchange()
    assert torch.equal(ex(h), h)
    flat = h.flatten(0, 1)
    assert torch.equal(ex.forward_packed(flat, conv_id_of((3,) * B)), flat)


# ---- packed (ragged, padding-free) layout ----

COUNTS = (2, 3)


def conv_id_of(counts):
    return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))


def make_packed_h(counts, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(int(sum(counts)), T, DIM, generator=gen)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_packed_zero_init_identity(kind):
    ex = make_exchange(kind)
    h = make_packed_h(COUNTS)
    with torch.no_grad():
        out = ex.forward_packed(h, conv_id_of(COUNTS))
    assert torch.equal(out, h)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_packed_equals_dense_per_conversation(kind):
    """A ragged packed batch matches each conversation run separately (dense,
    no padding), so conversations in one packed batch never mix."""
    ex = randomize(make_exchange(kind))
    h = make_packed_h(COUNTS, seed=3)
    with torch.no_grad():
        out = ex.forward_packed(h, conv_id_of(COUNTS))
        start = 0
        for n in COUNTS:
            dense = ex(h[start : start + n].unsqueeze(0))
            assert torch.allclose(out[start : start + n], dense[0], atol=1e-5)
            start += n


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_packed_permutation_equivariance(kind):
    """Rows may arrive in any order; only conv_id defines the groups."""
    ex = randomize(make_exchange(kind))
    h = make_packed_h(COUNTS, seed=5)
    cid = conv_id_of(COUNTS)
    perm = torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        out = ex.forward_packed(h, cid)
        out_perm = ex.forward_packed(h[perm], cid[perm])
    assert torch.allclose(out_perm, out[perm], atol=1e-5)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_packed_conv_id_shape_validation(kind):
    ex = make_exchange(kind)
    h = make_packed_h(COUNTS)
    with pytest.raises(ValueError):
        ex.forward_packed(h, torch.zeros(h.shape[0] + 1, dtype=torch.long))
