"""Branch-communication (exchange) modules.

Contract shared by every exchange in this file:

- Input ``(B, N, T, d)`` -> output ``(B, N, T, d)`` where ``N`` is the branch
  axis (one branch per speaker, all weights shared across branches).
- Permutation-equivariant in ``N`` and valid for any ``N >= 1``.
- NO positional encoding, index embedding, or any other branch-order-dependent
  computation on the branch axis: branches must be interchangeable.
- Optional ``pad_mask: (B, N)`` bool marks padded ghost branches
  (``True`` = padded); padded branches must not influence real ones.
- A scalar gate parameter ``g`` initialized to 0 makes the module exactly the
  identity at init (output bit-equal to input).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


class IdentityExchange(nn.Module):
    """No-communication baseline: returns its ``(B, N, T, d)`` input unchanged."""

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return h


class TACExchange(nn.Module):
    """Transform-average-concatenate exchange (Luo et al., ICASSP 2020) with a
    zero-init scalar gate.

    Maps ``(B, N, T, d) -> (B, N, T, d)`` per time frame:

    1. Transform: shared ``Linear(dim, hidden) + PReLU`` on every branch -> ``z_i``.
    2. Average: mean of ``z_i`` over the branch axis (excluding padded branches,
       divided by the real-branch count), then shared
       ``Linear(hidden, hidden) + PReLU`` -> ``z_bar``.
    3. Concatenate: per branch, ``Linear(2*hidden, dim) + PReLU`` on
       ``[z_i ; z_bar]`` -> ``u_i``.
    4. Output: ``h_i + g * u_i`` with ``g`` init 0 (exact identity at init).

    Permutation-equivariant in ``N`` (the branch average is order-free); no
    positional encoding anywhere on the branch axis.
    """

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = dim if hidden is None else hidden
        self.transform = nn.Sequential(nn.Linear(dim, hidden), nn.PReLU())
        self.average = nn.Sequential(nn.Linear(hidden, hidden), nn.PReLU())
        self.concat = nn.Sequential(nn.Linear(2 * hidden, dim), nn.PReLU())
        self.g = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.transform(h)  # (B, N, T, hidden)
        if pad_mask is None:
            z_bar = z.mean(dim=1, keepdim=True)
        else:
            keep = (~pad_mask).to(z.dtype)[:, :, None, None]  # (B, N, 1, 1)
            count = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
            z_bar = (z * keep).sum(dim=1, keepdim=True) / count
        z_bar = self.average(z_bar)  # (B, 1, T, hidden)
        u = self.concat(torch.cat((z, z_bar.expand_as(z)), dim=-1))
        return h + self.g * u


class BranchMHAExchange(nn.Module):
    """Multi-head self-attention over the branch axis with a zero-init scalar gate.

    Maps ``(B, N, T, d) -> (B, N, T, d)``:

    1. Pre-norm: shared ``LayerNorm(dim)``.
    2. Fold time into batch: ``(B, N, T, d) -> (B*T, N, d)`` so the branch axis
       is the attention sequence axis.
    3. MHA over the ``N`` branch tokens with shared projections
       ``W_q, W_k, W_v: dim -> d_c`` and ``W_o: d_c -> dim``; self-attention
       includes self; ``pad_mask`` acts as the key padding mask.
       NO positional encoding on the branch axis, so the module is
       permutation-equivariant in ``N``.
    4. Output: ``h_i + g * attn_out_i`` with ``g`` init 0 (exact identity at init).
    """

    def __init__(self, dim: int, n_heads: int = 8, d_c: int | None = None):
        super().__init__()
        d_c = dim if d_c is None else d_c
        if d_c % n_heads != 0:
            raise ValueError(f"d_c={d_c} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(dim)
        self.w_q = nn.Linear(dim, d_c)
        self.w_k = nn.Linear(dim, d_c)
        self.w_v = nn.Linear(dim, d_c)
        self.w_o = nn.Linear(d_c, dim)
        self.g = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, t, _ = h.shape
        x = rearrange(self.norm(h), "b n t d -> (b t) n d")
        q = rearrange(self.w_q(x), "bt n (nh e) -> bt nh n e", nh=self.n_heads)
        k = rearrange(self.w_k(x), "bt n (nh e) -> bt nh n e", nh=self.n_heads)
        v = rearrange(self.w_v(x), "bt n (nh e) -> bt nh n e", nh=self.n_heads)

        attn_mask = None
        if pad_mask is not None:
            # True = attend (SDPA convention); padded branches are masked out as
            # keys so they never influence real branches. Padded queries still
            # produce values, but those land only on padded output rows.
            attn_mask = repeat(~pad_mask, "b n -> (b t) 1 1 n", t=t)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "bt nh n e -> bt n (nh e)")
        out = rearrange(self.w_o(out), "(b t) n d -> b n t d", b=b)
        return h + self.g * out
