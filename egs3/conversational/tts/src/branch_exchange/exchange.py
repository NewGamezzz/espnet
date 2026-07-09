"""Branch-communication (exchange) modules.

Every exchange supports two input layouts:

- Packed ``forward_packed(h, conv_id, n_conv=None)``:
  ``(M, T, d) -> (M, T, d)`` where ``M = sum(N_i)`` stacks the branches of all
  conversations in the batch with NO padding rows at all. ``conv_id: (M,)``
  integer tensor gives each row's conversation (values in ``[0, n_conv)``);
  branches communicate only within their conversation. Rows need not be
  sorted or contiguous by conversation. This is the layout to train with:
  ragged speaker counts cost zero wasted compute or memory.
- Dense ``forward(h)``: ``(B, N, T, d) -> (B, N, T, d)`` where ``N`` is the
  branch axis (one branch per speaker), for batches where every conversation
  has the same speaker count.

The packed form is the core implementation; the dense form is a thin wrapper
(``conv_id = arange(B).repeat_interleave(N)``), so there is a single source
of truth for the math. There is deliberately NO padding/mask API: batches
with mixed speaker counts use the packed layout instead.

Contract shared by every exchange:

- Weights are shared across branches and conversations.
- Permutation-equivariant on the branch axis and valid for any count >= 1.
- NO positional encoding, index embedding, or any other branch-order-dependent
  computation on the branch axis: branches must be interchangeable.
- A scalar gate parameter ``g`` initialized to 0 makes the module exactly the
  identity at init (output bit-equal to input).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _check_conv_id(h: torch.Tensor, conv_id: torch.Tensor, n_conv: int | None) -> int:
    if conv_id.ndim != 1 or conv_id.shape[0] != h.shape[0]:
        raise ValueError(
            f"conv_id must have shape ({h.shape[0]},) matching the packed rows, "
            f"got {tuple(conv_id.shape)}"
        )
    if n_conv is None:
        n_conv = int(conv_id.max().item()) + 1 if conv_id.numel() else 0
    return n_conv


class IdentityExchange(nn.Module):
    """No-communication baseline: returns its input unchanged in both layouts."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def forward_packed(
        self, h: torch.Tensor, conv_id: torch.Tensor | None = None, n_conv: int | None = None
    ) -> torch.Tensor:
        return h


class _PackedExchange(nn.Module):
    """Base class implementing the dense ``(B, N, T, d)`` layout as a thin
    wrapper over the packed core ``forward_packed``.
    """

    def forward_packed(
        self, h: torch.Tensor, conv_id: torch.Tensor, n_conv: int | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n = h.shape[:2]
        conv_id = torch.arange(b, device=h.device).repeat_interleave(n)
        return self.forward_packed(h.flatten(0, 1), conv_id, n_conv=b).unflatten(0, (b, n))


class TACExchange(_PackedExchange):
    """Transform-average-concatenate exchange (Luo et al., ICASSP 2020) with a
    zero-init scalar gate.

    Packed core, per time frame on ``(M, T, d)`` rows grouped by ``conv_id``:

    1. Transform: shared ``Linear(dim, hidden) + PReLU`` on every branch -> ``z_i``.
    2. Average: segment mean of ``z_i`` over each conversation's branches
       (``index_add`` sum divided by the per-conversation branch count), then
       shared ``Linear(hidden, hidden) + PReLU`` -> ``z_bar``, gathered back to
       each row via ``conv_id``.
    3. Concatenate: per branch, ``Linear(2*hidden, dim) + PReLU`` on
       ``[z_i ; z_bar]`` -> ``u_i``.
    4. Output: ``h_i + g * u_i`` with ``g`` init 0 (exact identity at init).

    Permutation-equivariant on the branch axis (the segment mean is
    order-free); no positional encoding anywhere on the branch axis.
    """

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = dim if hidden is None else hidden
        self.transform = nn.Sequential(nn.Linear(dim, hidden), nn.PReLU())
        self.average = nn.Sequential(nn.Linear(hidden, hidden), nn.PReLU())
        self.concat = nn.Sequential(nn.Linear(2 * hidden, dim), nn.PReLU())
        self.g = nn.Parameter(torch.zeros(()))

    def forward_packed(
        self, h: torch.Tensor, conv_id: torch.Tensor, n_conv: int | None = None
    ) -> torch.Tensor:
        conv_id = conv_id.long()
        n_conv = _check_conv_id(h, conv_id, n_conv)
        z = self.transform(h)  # (M, T, hidden)
        z_sum = z.new_zeros((n_conv,) + z.shape[1:]).index_add_(0, conv_id, z)
        count = torch.bincount(conv_id, minlength=n_conv).clamp(min=1)
        z_bar = self.average(z_sum / count[:, None, None].to(z.dtype))  # (n_conv, T, hidden)
        u = self.concat(torch.cat((z, z_bar[conv_id]), dim=-1))
        return h + self.g * u


class BranchMHAExchange(_PackedExchange):
    """Multi-head self-attention over the branch axis with a zero-init scalar gate.

    Packed core on ``(M, T, d)`` rows grouped by ``conv_id``:

    1. Pre-norm: shared ``LayerNorm(dim)``.
    2. Fold time into batch: ``(M, T, d) -> (T, M, d)`` so the branch rows are
       the attention sequence axis.
    3. MHA over the ``M`` branch tokens with shared projections
       ``W_q, W_k, W_v: dim -> d_c`` and ``W_o: d_c -> dim``, restricted to a
       block-diagonal pattern (``conv_id[i] == conv_id[j]``) so branches only
       attend within their conversation; self-attention includes self.
       NO positional encoding on the branch axis, so the module is
       permutation-equivariant on it.
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

    def forward_packed(
        self, h: torch.Tensor, conv_id: torch.Tensor, n_conv: int | None = None
    ) -> torch.Tensor:
        conv_id = conv_id.long()
        _check_conv_id(h, conv_id, n_conv)
        x = rearrange(self.norm(h), "m t d -> t m d")
        q = rearrange(self.w_q(x), "t m (nh e) -> t nh m e", nh=self.n_heads)
        k = rearrange(self.w_k(x), "t m (nh e) -> t nh m e", nh=self.n_heads)
        v = rearrange(self.w_v(x), "t m (nh e) -> t nh m e", nh=self.n_heads)

        # True = attend (SDPA convention): block-diagonal same-conversation
        # mask, broadcast over the (T, heads) batch dims. Branches of other
        # conversations are masked out as keys, so they never mix.
        attn_mask = conv_id[:, None] == conv_id[None, :]  # (M, M)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "t nh m e -> t m (nh e)")
        out = rearrange(self.w_o(out), "t m d -> m t d")
        return h + self.g * out
