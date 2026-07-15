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
T, NT = 16, 8


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


def make_packed_inputs(counts, seed=0, t=T, nt=NT, mel_dim=MEL):
    """Packed DiT inputs: one row per branch, conversations stacked, no padding."""
    gen = torch.Generator().manual_seed(seed)
    m = int(sum(counts))
    x = torch.randn(m, t, mel_dim, generator=gen)
    cond = torch.randn(m, t, mel_dim, generator=gen)
    text = torch.randint(0, DIT_KWARGS["text_num_embeds"], (m, nt), generator=gen)
    time = torch.rand(m, generator=gen)
    return x, cond, text, time


def slice_conversation(inputs, counts, i):
    """Extract conversation ``i``'s rows from packed inputs: shapes (N_i, ...)."""
    start = int(sum(counts[:i]))
    end = start + counts[i]
    return tuple(t_[start:end] for t_ in inputs)


def inject_all(model, factory, depth=DEPTH):
    """Inject P+TAC at every block; returns the (inactive) BranchContext."""
    from branch_exchange import REGISTRY, BranchContext, ExchangeSchedule, inject_exchange

    ctx = BranchContext()
    schedule = ExchangeSchedule.from_spec({f"1-{depth}": "P+TAC"}, depth=depth, factory=factory)
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
