"""End-to-end smoke: 30 optimizer steps on synthetic data + EMA/deepcopy."""

import copy

import torch
from .conftest import MEL, T, make_packed_mels, randomize_params
from .test_build_model import build_tiny  # noqa: F401  (fixture reuse)

from egs3.conversational.tts.src.branch_exchange import ExchangedBlock, get_context
from egs3.conversational.tts.src.build_model import exchange_param_groups
from egs3.conversational.tts.src.lit_module import PackedConversationCollator


def _fake_samples(step: int, n: int = 2):
    """Fabricated ConversationDataset+preprocessor output (post-transform)."""
    gen = torch.Generator().manual_seed(1000 + step)
    samples = []
    for i, t_wav in enumerate((6144, 5120)):
        samples.append(
            {
                "window_id": f"w{step}_{i}",
                "num_channels": n,
                "speech": 0.1 * torch.randn(n, t_wav, generator=gen),
                "text": [torch.randint(0, 12, (30,), generator=gen) for _ in range(n)],
            }
        )
    return samples


def test_training_smoke(ext_vocab_file):
    torch.manual_seed(0)
    model = build_tiny(ext_vocab_file)
    # DiT zero-inits proj_out/AdaLN (no gradient reaches the blocks until
    # proj_out moves); the real model loads pretrained weights there, so
    # give the tiny stand-in non-zero weights too (gates stay zero).
    randomize_params(model, seed=42)
    model.train()
    optimizer = torch.optim.AdamW(
        exchange_param_groups(model, lr_exchange=1e-2, lr_backbone=1e-4)
    )
    collator = PackedConversationCollator()

    gates = [m.exchange.g for m in model.modules() if isinstance(m, ExchangedBlock)]
    for step in range(30):
        window_ids, batch = collator(_fake_samples(step))
        assert len(window_ids) == 2
        loss, stats, weight = model(**batch)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        assert torch.isfinite(stats["loss_ch0"]) and torch.isfinite(stats["loss_ch1"])
        assert int(weight) == 2  # conversations, not rows
        optimizer.zero_grad()
        loss.backward()
        if step == 0:
            grads = [g.grad for g in gates]
            assert all(grad is not None for grad in grads)
            assert any(grad.abs() > 0 for grad in grads)
        optimizer.step()

    assert any(g.detach().abs() > 0 for g in gates), "no gate moved off zero"


def test_ema_deepcopy_safety(ext_vocab_file):
    """copy.deepcopy of the assembled model succeeds and the copy's forward
    matches the original's (same inputs, eval mode)."""
    model = build_tiny(ext_vocab_file).eval()
    clone = copy.deepcopy(model).eval()

    # The copy's blocks share one NEW context, still consistent internally.
    assert get_context(clone.cfm.transformer) is clone.cfm.ctx
    assert clone.cfm.ctx is not model.cfm.ctx

    mel, text, lens = make_packed_mels([2], seed=5)
    gen = torch.Generator().manual_seed(6)
    kwargs = dict(
        counts=[2],
        lens=lens,
        frac_lengths=torch.tensor([0.8]),
        time=torch.tensor([0.5]),
        x0=torch.randn(2, T, MEL, generator=gen),
    )

    torch.manual_seed(7)  # span start draw
    loss1, _, extras1 = model.cfm(mel, text, **kwargs)
    torch.manual_seed(7)
    loss2, _, extras2 = clone.cfm(mel, text, **kwargs)

    assert torch.equal(loss1, loss2)
    assert torch.equal(extras1["pred"], extras2["pred"])
