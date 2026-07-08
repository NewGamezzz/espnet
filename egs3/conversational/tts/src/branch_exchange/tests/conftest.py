"""Shared path setup and helpers for the branch_exchange test suite."""

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
SRC_DIR = _HERE.parents[2]  # .../egs3/conversational/tts/src
REPO_ROOT = _HERE.parents[6]  # espnet repo root, so espnet2 resolves locally
for _p in (str(REPO_ROOT), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DIT_KWARGS = dict(dim=64, depth=4, heads=2, dim_head=32, mel_dim=20, text_num_embeds=50)
DIM = DIT_KWARGS["dim"]
DEPTH = DIT_KWARGS["depth"]
MEL = DIT_KWARGS["mel_dim"]
B, T, NT = 2, 16, 8


def make_dit(seed=0, **overrides):
    """Small random-init DiT in eval mode (dropout off, deterministic).

    DiT's own init zeroes ``proj_out`` and the AdaLN modulation layers, which
    would make every output exactly zero and starve the tests of signal, so we
    re-randomize all parameters with a seeded generator.
    """
    from espnet2.tts.f5.backbones.dit import DiT

    kwargs = dict(DIT_KWARGS)
    kwargs.update(overrides)
    torch.manual_seed(seed)
    model = DiT(**kwargs)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(0.02 * torch.randn(p.shape, generator=gen))
    model.eval()
    return model


def make_branch_inputs(n_branch, seed=0, batch=B, t=T, nt=NT, mel_dim=MEL):
    """Per-branch DiT inputs with a leading branch axis: shapes (B, N, ...)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n_branch, t, mel_dim, generator=gen)
    cond = torch.randn(batch, n_branch, t, mel_dim, generator=gen)
    text = torch.randint(
        0, DIT_KWARGS["text_num_embeds"], (batch, n_branch, nt), generator=gen
    )
    time = torch.rand(batch, n_branch, generator=gen)
    return x, cond, text, time


def run_independent(model, inputs):
    """Run the model separately on each branch's inputs and stack: (B, N, T, mel)."""
    x, cond, text, time = inputs
    outs = [
        model(x[:, i], cond[:, i], text[:, i], time[:, i]) for i in range(x.shape[1])
    ]
    return torch.stack(outs, dim=1)


def run_folded(model, inputs):
    """Fold the branch axis into batch, run once, unfold: (B, N, T, mel)."""
    x, cond, text, time = inputs
    out = model(
        x.flatten(0, 1), cond.flatten(0, 1), text.flatten(0, 1), time.flatten(0, 1)
    )
    return out.unflatten(0, (x.shape[0], x.shape[1]))


def inject_all(model, factory, depth=DEPTH):
    """Inject P+TAC at every block; returns the (inactive) BranchContext."""
    from branch_exchange import (
        REGISTRY,
        BranchContext,
        ExchangeSchedule,
        inject_exchange,
    )

    ctx = BranchContext()
    schedule = ExchangeSchedule.from_spec(
        {f"1-{depth}": "P+TAC"}, depth=depth, factory=factory
    )
    inject_exchange(model, REGISTRY["f5_dit"], schedule, ctx)
    return ctx


def iter_exchanges(model):
    from branch_exchange import BranchMHAExchange, TACExchange

    for m in model.modules():
        if isinstance(m, (TACExchange, BranchMHAExchange)):
            yield m


def set_gates(model, value):
    with torch.no_grad():
        for m in iter_exchanges(model):
            m.g.fill_(value)


def randomize_exchanges(model, seed=0):
    """Randomize all exchange weights except the gates (set those via set_gates)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in iter_exchanges(model):
            for name, p in m.named_parameters():
                if name == "g":
                    continue
                p.copy_(0.5 * torch.randn(p.shape, generator=gen))
