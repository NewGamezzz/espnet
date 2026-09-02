"""Module-level tests for the exchange contract on the packed layout
(M, T, d) + conv_id: zero-init identity, permutation equivariance,
branch-count generalization, and conversation isolation. No espnet
imports here."""

import pytest
import torch

from branch_exchange import BranchMHAExchange, IdentityExchange, TACExchange

DIM = 32
T = 5
COUNTS = (2, 3)


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


def conv_id_of(counts):
    return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))


def make_h(counts, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(int(sum(counts)), T, DIM, generator=gen)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_zero_init_identity(kind):
    ex = make_exchange(kind)
    h = make_h(COUNTS)
    with torch.no_grad():
        out = ex(h, conv_id_of(COUNTS))
    assert torch.equal(out, h)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_conversation_isolation(kind):
    """A ragged packed batch matches each conversation run on its own, so
    conversations in one batch never mix."""
    ex = randomize(make_exchange(kind))
    h = make_h(COUNTS, seed=3)
    with torch.no_grad():
        out = ex(h, conv_id_of(COUNTS))
        start = 0
        for n in COUNTS:
            alone = ex(h[start : start + n], torch.zeros(n, dtype=torch.long))
            assert torch.allclose(out[start : start + n], alone, atol=1e-5)
            start += n


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_permutation_equivariance(kind):
    """Rows may arrive in any order; only conv_id defines the groups."""
    ex = randomize(make_exchange(kind))
    h = make_h(COUNTS, seed=5)
    cid = conv_id_of(COUNTS)
    perm = torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        out = ex(h, cid)
        out_perm = ex(h[perm], cid[perm])
    assert torch.allclose(out_perm, out[perm], atol=1e-5)


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_any_branch_count(kind):
    ex = randomize(make_exchange(kind))
    for n in (1, 2, 3, 4):
        h = make_h((n,), seed=n)
        with torch.no_grad():
            out = ex(h, torch.zeros(n, dtype=torch.long))
        assert out.shape == h.shape


@pytest.mark.parametrize("kind", ["tac", "mha"])
def test_conv_id_shape_validation(kind):
    ex = make_exchange(kind)
    h = make_h(COUNTS)
    with pytest.raises(ValueError):
        ex(h, torch.zeros(h.shape[0] + 1, dtype=torch.long))


def test_identity_exchange():
    h = make_h(COUNTS)
    ex = IdentityExchange()
    assert torch.equal(ex(h, conv_id_of(COUNTS)), h)


# --- severed TAC: the capacity-matched no-communication control (Table 4 row c)


def test_severed_tac_defaults_off_and_changes_output_when_on():
    ex = randomize(TACExchange(DIM))
    assert ex.severed is False
    h = make_h(COUNTS, seed=5)
    with torch.no_grad():
        full = ex(h, conv_id_of(COUNTS))
        ex.severed = True
        cut = ex(h, conv_id_of(COUNTS))
    assert not torch.equal(full, cut)


def test_severed_tac_equals_every_branch_alone():
    """Severing == treating every branch as its own one-branch conversation:
    bit-identical to the unsevered module on conv_id = arange(M), which keeps
    the batch shape (and therefore the matmul reduction order) identical.

    Run branch-by-branch instead and the same identity holds only to ~1e-7,
    because a five-row matmul does not reduce in the same order as five
    one-row matmuls; that is arithmetic, not communication.
    """
    ex = randomize(TACExchange(DIM, severed=True))
    h = make_h(COUNTS, seed=7)
    with torch.no_grad():
        cut = ex(h, conv_id_of(COUNTS))
        ex.severed = False
        alone = ex(h, torch.arange(h.shape[0]))
        per_branch = torch.cat(
            [ex(h[i : i + 1], torch.zeros(1, dtype=torch.long)) for i in range(h.shape[0])]
        )
    assert torch.equal(cut, alone)
    assert torch.allclose(cut, per_branch, atol=1e-5)


def test_severed_tac_has_zero_cross_channel_gradient():
    ex = randomize(TACExchange(DIM, severed=True))
    h = make_h(COUNTS, seed=9).requires_grad_(True)
    conv_id = conv_id_of(COUNTS)
    # Row 3 is the second channel of the 3-channel conversation (rows 2, 3, 4).
    (grad,) = torch.autograd.grad(ex(h, conv_id)[3].sum(), h)
    assert torch.count_nonzero(grad[3]) > 0
    for j in (0, 1, 2, 4):
        assert torch.equal(grad[j], torch.zeros_like(grad[j])), j
    # Sanity: the same weights with the average live DO leak gradient.
    ex.severed = False
    (grad_full,) = torch.autograd.grad(ex(h, conv_id)[3].sum(), h)
    assert torch.count_nonzero(grad_full[2]) > 0 and torch.count_nonzero(grad_full[4]) > 0
    assert torch.equal(grad_full[0], torch.zeros_like(grad_full[0]))
