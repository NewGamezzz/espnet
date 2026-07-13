"""Shared fixtures for the multi-branch trainer test suite.

Fixture-based: no corpus and no pretrained checkpoint required (random-init
small DiT), pytest, CPU.  Tests import the code under test through its
package path (``egs3.conversational.tts.src``).
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[4]  # espnet repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egs3.conversational.tts.dataset.preprocessing.text import (  # noqa: E402
    NEW_TOKENS,
)
from egs3.conversational.tts.src.branch_exchange import (  # noqa: E402
    REGISTRY,
    BranchContext,
    ExchangedBlock,
    ExchangeSchedule,
    TACExchange,
    inject_exchange,
)
from egs3.conversational.tts.src.multibranch_cfm import MultiBranchCFM  # noqa: E402
from espnet2.tts.f5.backbones.dit import DiT  # noqa: E402

MEL = 20
T, NT = 24, 10
TINY_ARCH = dict(
    dim=64,
    depth=4,
    heads=2,
    dim_head=32,
    text_dim=32,
    conv_layers=2,
    dropout=0.0,  # keep forward deterministic in train mode too
    text_mask_padding=False,
    pe_attn_head=1,
)
CFM_KWARGS = dict(
    sigma=0.0,
    # No CFG drops: the equivalence tests need bit-identical conditioning.
    audio_drop_prob=0.0,
    cond_drop_prob=0.0,
    frac_lengths_mask=(0.7, 1.0),
    num_channels=MEL,
)
FEATS_KWARGS = dict(
    fs=24000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=MEL,
    mel_spec_type="vocos",
)

# base tokens include " " (the surgery's warm-start source for <turn>).
BASE_TOKENS = [" "] + list(string.ascii_lowercase[:10])
EXT_TOKENS = BASE_TOKENS + list(NEW_TOKENS)


@pytest.fixture
def ext_vocab_file(tmp_path) -> Path:
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(EXT_TOKENS) + "\n", encoding="utf-8")
    return path


def randomize_params(module: torch.nn.Module, seed: int) -> None:
    """Seeded re-randomization: DiT's own init zeroes proj_out/AdaLN, which
    would starve the tests of signal.  Injected gates are re-zeroed so the
    zero-init identity is preserved."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(0.02 * torch.randn(p.shape, generator=gen))
        for m in module.modules():
            if isinstance(m, ExchangedBlock):
                m.exchange.g.zero_()


def make_dit(seed: int = 0, text_num_embeds: int = len(EXT_TOKENS)) -> DiT:
    torch.manual_seed(seed)
    dit = DiT(mel_dim=MEL, text_num_embeds=text_num_embeds, **TINY_ARCH)
    randomize_params(dit, seed + 1)
    return dit


def make_multibranch(dit: DiT, schedule_spec=None) -> MultiBranchCFM:
    """Wrap ``dit`` into a MultiBranchCFM with TAC injected per the spec."""
    ctx = BranchContext()
    cfm = MultiBranchCFM(dit, ctx=ctx, **CFM_KWARGS)
    schedule = ExchangeSchedule.from_spec(
        schedule_spec or {f"1-{TINY_ARCH['depth']}": "P+TAC"},
        depth=TINY_ARCH["depth"],
        factory=lambda: TACExchange(TINY_ARCH["dim"]),
    )
    inject_exchange(dit, REGISTRY["f5_dit"], schedule, ctx)
    return cfm


def make_packed_mels(counts, seed: int = 0, t: int = T, nt: int = NT):
    """Packed mel/text rows for the given per-conversation counts."""
    gen = torch.Generator().manual_seed(seed)
    rows = int(sum(counts))
    mel = torch.randn(rows, t, MEL, generator=gen)
    text = torch.randint(0, len(EXT_TOKENS), (rows, nt), generator=gen)
    lens = torch.full((rows,), t, dtype=torch.long)
    return mel, text, lens


def deterministic_span_mask(seq_len: torch.Tensor, frac_lengths: torch.Tensor):
    """mask_from_frac_lengths with the span start pinned to the middle,
    so reference CFM and MultiBranchCFM see identical spans without
    coordinating their RNG draw order."""
    from espnet2.tts.f5.utils import mask_from_start_end_indices

    lengths = (frac_lengths * seq_len).long()
    start = ((seq_len - lengths) // 2).long()
    return mask_from_start_end_indices(seq_len, start, start + lengths)
