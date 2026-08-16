"""MultiBranchCFM: zero-init equivalence and shared span/time sampling."""

import copy

import pytest
import torch
from .conftest import (
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


def test_deterministic_mask_covers_target_exactly():
    """cond_frames >= 0 pins the span mask to exactly [cond_frames, conv_len)
    - the whole target region, no randomness."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=7, t=40)

    _, _, extras = multibranch(
        mel,
        text,
        counts=[2],
        lens=lens,
        cond_frames=torch.tensor([12]),
        time=torch.tensor([0.5]),
    )

    m = extras["rand_span_mask"]
    assert not m[:, :12].any()
    assert m[:, 12:40].all()


def test_sentinel_batch_bit_identical():
    """An all-sentinel cond_frames=[-1] batch must consume identical RNG and
    produce a bit-identical loss/mask to omitting the kwarg entirely - the
    random frac_lengths draw stays unconditional."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=9, t=40)

    torch.manual_seed(0)
    ref_loss, _, ref_extras = multibranch(mel, text, counts=[2], lens=lens)

    torch.manual_seed(0)
    new_loss, _, new_extras = multibranch(
        mel, text, counts=[2], lens=lens, cond_frames=torch.tensor([-1])
    )

    torch.testing.assert_close(ref_loss, new_loss)
    assert torch.equal(ref_extras["rand_span_mask"], new_extras["rand_span_mask"])


def test_mixed_batch_routes_per_conversation():
    """Two conversations in one batch: conv 0 deterministic (cond_frames=12),
    conv 1 random (cond_frames=-1, frac_lengths=0.5) - each conversation's
    rows share their own span, conv 0's is overridden, and conv 1's is
    routed through UNTOUCHED. That second half is the actual routing test:
    a broken torch.where(det.any(), det_mask, conv_span_mask) (clobbering
    every conversation once ANY conversation is deterministic) would still
    pass an assertion that only inspects conv 0, so re-run the identical
    call with cond_frames=None from the same seed and require conv 1's rows
    to be bit-identical across the two runs."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2, 2], seed=11, t=40)
    common = dict(
        counts=[2, 2],
        lens=lens,
        frac_lengths=torch.tensor([0.0, 0.5]),
        time=torch.tensor([0.5, 0.5]),
    )

    torch.manual_seed(21)
    _, _, extras_det = multibranch(
        mel, text, cond_frames=torch.tensor([12, -1]), **common
    )
    torch.manual_seed(21)
    _, _, extras_rand = multibranch(mel, text, **common)

    m_det = extras_det["rand_span_mask"]
    m_rand = extras_rand["rand_span_mask"]

    # conv 0 (rows 0-1): overridden to the deterministic span exactly [12, 40).
    assert torch.equal(m_det[0], m_det[1])
    assert not m_det[0, :12].any()
    assert m_det[0, 12:].all()

    # conv 1 (rows 2-3): sentinel, must be routed through untouched - bit
    # identical to the cond_frames=None run, and shared within the conv.
    assert torch.equal(m_det[2], m_rand[2])
    assert torch.equal(m_det[3], m_rand[3])
    assert torch.equal(m_det[2], m_det[3])


def test_cond_frames_length_mismatch_raises():
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2, 2], seed=13, t=40)
    with pytest.raises(ValueError, match="cond_frames"):
        multibranch(mel, text, counts=[2, 2], lens=lens, cond_frames=torch.tensor([12]))


def test_deterministic_mask_empty_span_raises():
    """cond_frames >= conv_len collapses the deterministic span to empty,
    which would otherwise silently NaN the masked-mean loss - fail loudly
    instead (same house style as the share-one-length guard above)."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=15, t=40)
    with pytest.raises(ValueError, match="empty"):
        multibranch(mel, text, counts=[2], lens=lens, cond_frames=torch.tensor([40]))


def test_sample_seed_reproducible_and_channels_independent():
    """A fixed seed reproduces the run bit-exactly, but must NOT give the
    channels identical noise (CFM.sample re-seeds per row; MultiBranchCFM
    seeds once so rows draw sequentially)."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()

    gen = torch.Generator().manual_seed(11)
    # Identical cond and text on both channels: any output difference can
    # only come from per-channel noise.
    cond = torch.randn(1, T, MEL, generator=gen).repeat(2, 1, 1)
    text = torch.randint(0, 12, (1, 8), generator=gen).repeat(2, 1)
    lens = torch.full((2,), 4, dtype=torch.long)  # 4-frame prompt

    def run():
        out, _ = multibranch.sample(
            cond,
            text,
            duration=T,
            counts=[2],
            lens=lens,
            steps=2,
            cfg_strength=0.0,
            seed=123,
        )
        return out

    out1, out2 = run(), run()
    assert torch.equal(out1, out2)  # reproducible
    generated = out1[:, 4:, :]  # past the prompt region
    assert not torch.equal(generated[0], generated[1])  # independent noise


def test_mask_kwargs_all_false_bit_identical():
    """All-False context_rows/independent_mask must consume identical RNG and
    produce bit-identical loss/mask to omitting the kwargs (same guarantee as
    the cond_frames sentinel)."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=9, t=40)

    torch.manual_seed(0)
    ref_loss, _, ref_extras = multibranch(mel, text, counts=[2], lens=lens)

    torch.manual_seed(0)
    new_loss, _, new_extras = multibranch(
        mel,
        text,
        counts=[2],
        lens=lens,
        context_rows=torch.tensor([False, False]),
        independent_mask=torch.tensor([False]),
    )
    torch.testing.assert_close(ref_loss, new_loss)
    assert torch.equal(
        ref_extras["rand_span_mask"], new_extras["rand_span_mask"]
    )


def test_context_rows_fully_observed_and_loss_excluded():
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=3, t=40)
    loss, stats, extras = multibranch(
        mel,
        text,
        counts=[2],
        lens=lens,
        context_rows=torch.tensor([True, False]),
    )
    # Context row: never masked -> cond carries the full ground-truth mel.
    assert not extras["rand_span_mask"][0].any()
    torch.testing.assert_close(extras["cond"][0], mel[0])
    # Zero loss frames on ch0 -> its stats key is skipped, ch1's is present.
    assert "loss_ch0" not in stats
    assert "loss_ch1" in stats
    assert torch.isfinite(loss)


def test_context_all_rows_of_a_conversation_raises():
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=3, t=40)
    with pytest.raises(ValueError, match="every row"):
        multibranch(
            mel,
            text,
            counts=[2],
            lens=lens,
            context_rows=torch.tensor([True, True]),
        )


def test_mask_kwarg_shape_mismatches_raise():
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=3, t=40)
    with pytest.raises(ValueError, match="context_rows"):
        multibranch(
            mel, text, counts=[2], lens=lens,
            context_rows=torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="independent_mask"):
        multibranch(
            mel, text, counts=[2], lens=lens,
            independent_mask=torch.tensor([False, False]),
        )


def test_independent_rows_draw_their_own_spans(monkeypatch):
    """With the span start pinned, per-row frac_lengths of different sizes
    must yield per-row masks of different lengths."""
    monkeypatch.setattr(
        mb_mod, "mask_from_frac_lengths", deterministic_span_mask
    )
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=5, t=40)
    _, _, extras = multibranch(
        mel,
        text,
        counts=[2],
        lens=lens,
        independent_mask=torch.tensor([True]),
        row_frac_lengths=torch.tensor([0.3, 0.8]),
    )
    sums = extras["rand_span_mask"].sum(dim=1)
    assert sums[0] != sums[1]


def test_context_target_rows_draw_their_own_spans(monkeypatch):
    """Context conversations' non-context (target) rows also get per-row
    frac draws - distinct from the shared conversation-level span."""
    monkeypatch.setattr(
        mb_mod, "mask_from_frac_lengths", deterministic_span_mask
    )
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([3], seed=5, t=40)
    _, _, extras = multibranch(
        mel,
        text,
        counts=[3],
        lens=lens,
        context_rows=torch.tensor([True, False, False]),
        row_frac_lengths=torch.tensor([0.0, 0.3, 0.8]),
    )
    sums = extras["rand_span_mask"].sum(dim=1)
    assert sums[0] == 0
    assert sums[1] != sums[2]


def test_independent_is_noop_on_chunk_conversations():
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=5, t=40)
    torch.manual_seed(0)
    _, _, ref = multibranch(
        mel, text, counts=[2], lens=lens, cond_frames=torch.tensor([12])
    )
    torch.manual_seed(0)
    _, _, new = multibranch(
        mel,
        text,
        counts=[2],
        lens=lens,
        cond_frames=torch.tensor([12]),
        independent_mask=torch.tensor([True]),
    )
    assert torch.equal(ref["rand_span_mask"], new["rand_span_mask"])


def test_chunk_context_composition():
    """Chunk x context: the context row is observed past cond_frames, the
    target row keeps the deterministic [cond_frames, len) span."""
    multibranch = make_multibranch(make_dit(seed=0)).eval()
    mel, text, lens = make_packed_mels([2], seed=5, t=40)
    _, _, extras = multibranch(
        mel,
        text,
        counts=[2],
        lens=lens,
        cond_frames=torch.tensor([12]),
        context_rows=torch.tensor([True, False]),
    )
    span = extras["rand_span_mask"]
    assert not span[0].any()
    expected = torch.zeros_like(span[1])
    expected[12 : int(lens[1])] = True
    assert torch.equal(span[1], expected)
