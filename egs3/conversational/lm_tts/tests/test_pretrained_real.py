"""Zero-init parity against the REAL BagPiper (speechlm-qwen3-8b) checkpoint.

The unit suite (``test_qwen3_injection.py``) proves zero-gate identity on a
tiny random-init Qwen3; this test closes the remaining gap by running the
actual production checkpoint through the same invariant: injecting
``TACExchange`` at zero-init gates into the real 36-layer, 4096-hidden model
must not change a single bit of the forward output on identical inputs.

Skipped unless ``BAGPIPER_CKPT`` (the safetensors shard DIRECTORY, e.g.
``downloads/bagpiper/speechlm-qwen3-8b``) is set and exists.
``BAGPIPER_TRAIN_CONFIG`` defaults to the committed
``conf/bagpiper_train_config.yaml`` when unset.

CPU/bf16 only: the retained model is ~16.9 GB in bf16 and this repo's dev
machine has 16 GiB RAM, so this test can only actually run on a bigger box
(see docs/bagpiper-findings.md "Gate results"). Zero-gate identity
(``h + 0*u == h``) is exact bitwise in any dtype, so ``torch.equal`` still
holds under bf16.
"""

import os

import pytest
import torch

from src.branch_exchange import BranchContext, ExchangeSchedule, TACExchange, inject_exchange
from src.branch_exchange.registry import REGISTRY

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get("BAGPIPER_CKPT")
CFG = os.environ.get("BAGPIPER_TRAIN_CONFIG") or os.path.join(
    RECIPE_DIR, "conf", "bagpiper_train_config.yaml"
)

pytestmark = pytest.mark.skipif(
    not (CKPT and os.path.exists(CKPT)),
    reason=(
        "set BAGPIPER_CKPT to the BagPiper safetensors shard directory to run "
        "real-checkpoint parity (BAGPIPER_TRAIN_CONFIG defaults to "
        "conf/bagpiper_train_config.yaml)"
    ),
)


def _duplicate_batch(batch: dict, n: int) -> dict:
    """Repeat a real 1-row preprocessor batch into ``n`` identical rows.

    Matches the packed-batch contract ``ctx.branches(counts=[n])`` expects
    ((sum(counts), T, d) hidden states, no padding rows): every tensor field
    is repeated along the batch dim, except ``discrete_audio_indices`` (rows
    of ``(bidx, start, length)``) whose ``bidx`` column must become
    ``0..n-1`` rather than repeating the same index ``n`` times, so
    ``_embed``'s ``zip(codes, io_indices)`` still lines each duplicated
    audio's decoded codes up with its own batch row.
    """
    out = {}
    for key, value in batch.items():
        if key == "discrete_audio_indices":
            rows = []
            for row in range(n):
                seg = value.clone()
                seg[:, 0] = row
                rows.append(seg)
            out[key] = torch.cat(rows, dim=0)
        elif isinstance(value, torch.Tensor):
            out[key] = value.repeat(n, *([1] * (value.dim() - 1)))
        else:
            out[key] = value
    return out


def _model_forward_hidden_states(model, batch: dict) -> torch.Tensor:
    """Run the model's real training-forward internals, returning the
    per-stream hidden state (pre ``lm_head``, pre-loss) instead of the scalar
    loss.

    ``ParallelLLM.forward`` (espnet2/speechlm/model/speechlm/lm/parallel.py)
    always calls ``self._loss(...)``, which asserts on ``loss_mask.size()``
    and only ever returns ``{"loss", "stats"}`` -- there is no logits-only
    branch to call into. This wrapper instead reproduces ``forward``'s
    embedding + backbone + stream-embedding steps (identical to both the
    "loss" path and the ``_step`` inference path) and stops there.

    Deliberately NOT full-vocab logits: BagPiper's vocab is 160392, and this
    forward's per-stream hidden state broadcasts to
    ``(batch, seq, 8, 4096)``; running that through ``lm_head`` would
    materialize ``(2, 2511, 8, 160392)`` bf16 logits per call -- about
    12.9 GB, ~26 GB held simultaneously for ``torch.equal``'s base-vs-injected
    comparison, on top of the ~16.9 GB model. That is exactly the blowup
    ``_loss``'s interval-based softmax exists to avoid (see
    docs/bagpiper-findings.md "Teacher-forced loss" gate item), and risks OOM
    even on a box sized for that constraint. ``lm_head`` is a fixed matmul
    independent of the TAC injection (which only perturbs the hidden-state
    pathway feeding it), so hidden-state bit-exactness implies logit
    bit-exactness a fortiori -- this is exactly as rigorous a parity proof and
    about 40x cheaper. Both the base and TAC-injected calls in this test go
    through this SAME function on identical inputs, so a zero gate must
    produce a bit-identical result.
    """
    input_ids = batch["seqs"].clone()
    position_ids = batch.get("position_ids")
    inputs_embeds = model._embed(input_ids, batch)
    output = model.model(inputs_embeds=inputs_embeds, position_ids=position_ids)
    hidden_states = output.last_hidden_state.unsqueeze(2)
    stream_emb = model.stream_emb.weight.tile(1, 1, 1, 1)
    stream_emb[:, :, 0] = 0.0  # first stream uses the base representation
    hidden_states = hidden_states + stream_emb
    return hidden_states


def test_zero_init_parity_real_checkpoint():
    import sys

    sys.path.insert(0, os.path.join(RECIPE_DIR, "scripts"))
    import yaml
    from gate_teacher_forced import build_batch  # noqa: E402
    from load_bagpiper import load_bagpiper  # noqa: E402

    model = load_bagpiper(CFG, CKPT, device="cpu", dtype=torch.bfloat16)

    with open(CFG) as f:
        train_config = yaml.safe_load(f)
    batch, _example_id = build_batch(train_config)
    batch = _duplicate_batch(batch, n=2)

    torch.manual_seed(0)
    with torch.no_grad():
        base = _model_forward_hidden_states(model, batch)

    ctx = BranchContext()
    schedule = ExchangeSchedule.from_spec(
        {"1-18": "P", "19-36": "P+TAC"}, depth=36, factory=lambda: TACExchange(4096)
    )
    inject_exchange(model, REGISTRY["qwen3"], schedule, ctx)

    with torch.no_grad(), ctx.branches(counts=[2]):
        injected = _model_forward_hidden_states(model, batch)

    assert torch.equal(base, injected)  # zero gate => bit-exact
