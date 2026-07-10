"""MultiBranchCFM: zero-init equivalence and shared span/time sampling."""

import copy

import pytest
import torch
from conftest import (
    CFM_KWARGS,
    MEL,
    T,
    deterministic_span_mask,
    make_dit,
    make_multibranch,
    make_packed_mels,
)

import egs3.conversational.tts.src.multibranch_cfm as mb_mod
import espnet2.tts.f5.cfm as cfm_mod
from espnet2.tts.f5.cfm import CFM


@pytest.mark.parametrize("n", [2, 3])
def test_equivalence_at_init(monkeypatch, n):
    """With zero-init gates and fixed span/time/noise, the multi-branch loss
    equals the average of N reference CFM losses on the individual channels.

    The span start is pinned (both modules patched to the same deterministic
    ``mask_from_frac_lengths``) so the reference's internal RNG draw order
    can be replayed exactly: uniform_ (frac), randn_like (x0), rand (time).
    """
    monkeypatch.setattr(cfm_mod, "mask_from_frac_lengths", deterministic_span_mask)
    monkeypatch.setattr(mb_mod, "mask_from_frac_lengths", deterministic_span_mask)

    dit = make_dit(seed=0)
    reference = CFM(transformer=copy.deepcopy(dit), **CFM_KWARGS).eval()
    multibranch = make_multibranch(dit).eval()

    mel, text, lens = make_packed_mels([n], seed=7)
    seed = 123

    # Replay the reference's internal sampling (all channels use the same
    # seed, which is exactly the shared-span/shared-time constraint).
    torch.manual_seed(seed)
    frac = torch.zeros(1).float().uniform_(*CFM_KWARGS["frac_lengths_mask"])
    x0_row = torch.randn(1, T, MEL)
    time = torch.rand(1)

    reference_losses = []
    for k in range(n):
        torch.manual_seed(seed)
        loss_k, _, _ = reference(mel[k : k + 1], text[k : k + 1], lens=lens[k : k + 1])
        reference_losses.append(loss_k)

    loss, stats, _ = multibranch(
        mel,
        text,
        counts=[n],
        lens=lens,
        frac_lengths=frac,
        time=time,
        x0=x0_row.repeat(n, 1, 1),
    )

    expected = torch.stack(reference_losses).mean()
    assert torch.allclose(loss, expected, rtol=1e-4, atol=1e-6)
    # Per-channel stats line up with the per-channel reference runs.
    for k in range(n):
        assert torch.allclose(
            stats[f"loss_ch{k}"], reference_losses[k], rtol=1e-4, atol=1e-6
        )


@pytest.mark.parametrize("counts", [[2, 3], [3, 2]])
def test_shared_sampling(counts):
    """Span mask and flow time are identical across the N rows of each
    conversation and differ across conversations."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels(counts, seed=1)
    # Distinct per-conversation lengths (rows within a conversation share).
    lens[counts[0] :] = T - 4
    mel[counts[0] :, T - 4 :, :] = 0.0

    torch.manual_seed(0)
    _, _, extras = multibranch(mel, text, counts=counts, lens=lens)

    span = extras["rand_span_mask"]
    row = 0
    for n in counts:
        for k in range(1, n):
            assert torch.equal(span[row], span[row + k])
        row += n
    assert not torch.equal(span[0], span[counts[0]])

    time = extras["time"]
    assert time.shape == (len(counts),)
    assert time[0] != time[1]
    frac = extras["frac_lengths"]
    assert frac.shape == (len(counts),)
    assert frac[0] != frac[1]


def test_counts_mismatch_raises():
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=2)
    with pytest.raises(ValueError, match="sum\\(counts\\)"):
        multibranch(mel, text, counts=[3], lens=lens)


def test_unequal_lens_within_conversation_raises():
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=3)
    lens[1] = T - 2
    with pytest.raises(ValueError, match="share one length"):
        multibranch(mel, text, counts=[2], lens=lens)
