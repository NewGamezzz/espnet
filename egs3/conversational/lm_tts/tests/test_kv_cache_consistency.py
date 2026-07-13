import torch

from src.branch_exchange import BranchContext, ExchangeSchedule, TACExchange, inject_exchange
from src.branch_exchange.registry import REGISTRY


def _inject_open_gate(model, dim=64):
    ctx = BranchContext()
    schedule = ExchangeSchedule.from_spec(
        {"1-2": "P", "3-4": "P+TAC"}, depth=4, factory=lambda: TACExchange(dim)
    )
    inject_exchange(model, REGISTRY["qwen3"], schedule, ctx)
    torch.manual_seed(1)
    for m in model.modules():
        if type(m).__name__ == "ExchangedBlock":
            torch.nn.init.normal_(m.exchange.g)  # open the gate: exchange really mixes
    return ctx


def test_stepwise_decode_matches_teacher_forced_equal_prefix(tiny_qwen3):
    """TAC is per-frame, so stepwise decode with cache must equal the full forward."""
    ctx = _inject_open_gate(tiny_qwen3)
    p, L = 6, 5
    ids = torch.randint(0, 128, (2, p + L))
    offsets = torch.tensor([p, p])
    with torch.no_grad():
        with ctx.branches(counts=[2], align_offsets=offsets, align_len=L):
            full = tiny_qwen3(ids).logits  # teacher-forced, offset-aligned exchange
        past = tiny_qwen3(ids[:, :p], use_cache=True).past_key_values  # prefill: ctx inactive
        step_logits = []
        with ctx.branches(counts=[2]):  # decode: active, plain per-step path
            for k in range(L):
                out = tiny_qwen3(ids[:, p + k : p + k + 1], past_key_values=past, use_cache=True)
                past = out.past_key_values
                step_logits.append(out.logits[:, 0])
    for k in range(L):
        assert torch.allclose(full[:, p + k], step_logits[k], atol=1e-4), f"audio step {k}"


def test_stepwise_decode_matches_teacher_forced_mixed_prefix(tiny_qwen3):
    """Mixed prefix lengths: right-padded teacher-forced (offsets) vs left-padded stepwise."""
    ctx = _inject_open_gate(tiny_qwen3)
    p0, p1, L = 4, 6, 5
    torch.manual_seed(2)
    pre0, pre1 = torch.randint(0, 128, (p0,)), torch.randint(0, 128, (p1,))
    audio = torch.randint(0, 128, (2, L))
    # teacher-forced: right-pad row 0 at the END, audio at true offsets
    tf_ids = torch.zeros(2, p1 + L, dtype=torch.long)
    tf_ids[0, :p0], tf_ids[0, p0 : p0 + L] = pre0, audio[0]
    tf_ids[1, :p1], tf_ids[1, p1 : p1 + L] = pre1, audio[1]
    tf_mask = torch.ones(2, p1 + L, dtype=torch.long)
    tf_mask[0, p0 + L :] = 0
    tf_pos = (tf_mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        with ctx.branches(counts=[2], align_offsets=torch.tensor([p0, p1]), align_len=L):
            full = tiny_qwen3(tf_ids, attention_mask=tf_mask, position_ids=tf_pos).logits
        # stepwise: LEFT-pad prefixes so audio starts aligned, then decode L steps
        lp_ids = torch.zeros(2, p1, dtype=torch.long)
        lp_ids[0, p1 - p0 :], lp_ids[1] = pre0, pre1
        lp_mask = torch.zeros(2, p1, dtype=torch.long)
        lp_mask[0, p1 - p0 :], lp_mask[1] = 1, 1
        lp_pos = (lp_mask.cumsum(-1) - 1).clamp(min=0)
        out = tiny_qwen3(lp_ids, attention_mask=lp_mask, position_ids=lp_pos, use_cache=True)
        past, mask = out.past_key_values, lp_mask
        step_logits = []
        with ctx.branches(counts=[2]):
            for k in range(L):
                mask = torch.cat([mask, torch.ones(2, 1, dtype=torch.long)], dim=-1)
                pos = (mask.sum(-1, keepdim=True) - 1)
                out = tiny_qwen3(
                    audio[:, k : k + 1], attention_mask=mask,
                    position_ids=pos, past_key_values=past, use_cache=True,
                )
                past = out.past_key_values
                step_logits.append(out.logits[:, 0])
    tf_audio_logit_pos = [(p0 + k, p1 + k) for k in range(L)]
    for k, (r0, r1) in enumerate(tf_audio_logit_pos):
        assert torch.allclose(full[0, r0], step_logits[k][0], atol=1e-4), f"row 0 step {k}"
        assert torch.allclose(full[1, r1], step_logits[k][1], atol=1e-4), f"row 1 step {k}"
