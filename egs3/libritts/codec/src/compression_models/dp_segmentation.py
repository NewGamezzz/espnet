import math
from typing import Optional

import torch

from .base import BaseCompressionModel, CompressionOutput


class DPSegmentationCompression(BaseCompressionModel):
    """Optimal frame boundary detection via Dynamic Programming.

    Based on CodecSlime (arXiv:2506.21074).  Finds the segmentation that
    minimises total L2 distortion between original frames and their
    segment-mean reconstruction:

        L(j, s) = Σ_{t=j−s+1}^{j} ‖h_t − mean(h[j−s+1 … j])‖₂

    The DP maximises the cumulative negative distortion:

        d[j, i] = max_{1 ≤ s ≤ T}  { d[j−s, i−1] − L(j, s) }
        d[0, 0] = 0

    targeting exactly T' = clamp(round(rate × T), 1, T) segments.

    Optimal boundaries are recovered by backtracking through stored parent
    pointers.  The cost table and DP run under ``torch.no_grad()`` because
    boundaries are hard (non-differentiable); gradients still flow through
    the subsequent segment-averaging step.

    Args:
        max_span: Hard upper bound on segment length U (default 4, matching
                  the paper).  This is an **algorithmic parameter** — it is
                  part of the formal DP constraint set
                  ``S = {s | Σsᵢ = T, 1 ≤ sᵢ ≤ U}`` and should be kept
                  consistent between training and inference.  Larger values
                  allow more aggressive compression but increase cost-table
                  memory from O(T·U·D) toward O(T²·D).  Set to ``None``
                  to remove the constraint entirely (equivalent to U = T).

    Note — variable-rate inference:
        The paper's model is "schedulable": R_S (the compression rate) is
        passed as a runtime argument to the DP scheduler, so a single trained
        model can be evaluated at multiple rates at inference time.  Our
        ``rate`` argument mirrors this exactly.  U, however, is fixed at
        training time and should not be changed at inference.
    """

    def __init__(
        self,
        max_span: Optional[int] = 4,
        percent_kept_boundary: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.max_span = max_span
        # Anchor support is "all-or-none" by design (Option A): when enabled,
        # every anchor is kept as a forced boundary.  percent_kept_boundary acts
        # as a switch — > 0 enables anchors, 0/None disables.  The fractional
        # value is preserved for API compatibility with cosine_similarity but
        # has no effect on selection (anchors are not subsampled).
        self.percent_kept_boundary = percent_kept_boundary

    # ------------------------------------------------------------------
    # Cost-table computation (vectorised over windows for each span s)
    # ------------------------------------------------------------------

    def _compute_cost_table(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-compute L2 distortion for every (end-frame, span) pair.

        ``cost[j, s−1]`` is the cost of a segment of length *s* whose
        last frame is 0-indexed frame *j*:

            cost[j, s−1] = Σ_{t=j−s+1}^{j} ‖h_t − mean(h[j−s+1 … j])‖₂

        Invalid entries (j < s−1) are set to ``+inf``.

        Args:
            x: (T, D) unpadded feature sequence.
        Returns:
            cost: (T, U) where U = min(max_span, T) or T when max_span is
                  None; invalid entries are ``+inf``.
        """
        T, D = x.shape
        U = min(self.max_span, T) if self.max_span is not None else T
        device, dtype = x.device, x.dtype

        cost = torch.full((T, U), float("inf"), device=device, dtype=dtype)

        for s in range(1, U + 1):
            n_windows = T - s + 1
            if n_windows <= 0:
                break

            # Build sliding-window index: idx[w, t] = w + t
            w_idx = torch.arange(n_windows, device=device)  # (n_windows,)
            t_idx = torch.arange(s, device=device)  # (s,)
            idx = w_idx.unsqueeze(1) + t_idx.unsqueeze(0)  # (n_windows, s)

            frames = x[idx]  # (n_windows, s, D)
            seg_mean = frames.mean(dim=1, keepdim=True)  # (n_windows, 1, D)
            l2_sum = (frames - seg_mean).norm(dim=-1).sum(dim=1)  # (n_windows,)

            # 0-indexed end frames for windows of length s: s−1, s, …, T−1
            j_vals = torch.arange(s - 1, T, device=device)  # (n_windows,)
            cost[j_vals, s - 1] = l2_sum

        return cost  # (T, U)

    # ------------------------------------------------------------------
    # DP forward pass + backtracking
    # ------------------------------------------------------------------

    def _dp_forward(
        self,
        cost: torch.Tensor,
        T: int,
        max_segments: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward DP pass over (j, i) states up to ``max_segments``.

        Returns:
            d:      (T+1, max_segments+1) — d[j, i] is the best objective
                    (negative distortion) covering j frames with i segments.
                    NEG_INF for infeasible (j, i) combinations.
            parent: (T+1, max_segments+1) — segment length s chosen to reach
                    state (j, i); 1-indexed span.
        """
        U = cost.shape[1]
        device = cost.device
        NEG_INF = float("-inf")

        d = torch.full(
            (T + 1, max_segments + 1), NEG_INF, device=device, dtype=cost.dtype
        )
        d[0, 0] = 0.0

        parent = torch.zeros((T + 1, max_segments + 1), dtype=torch.long, device=device)

        # For span s_idx (0-indexed, span length = s_idx+1):
        #   candidates[s_idx, j] = d_prev[j - s_idx] - cost[j, s_idx]
        # Build a padded view of d_prev so shifting is a plain slice:
        #   padded = [NEG_INF×U, d_prev[0..T-1]]  (length U+T)
        #   padded[U - s_idx + j] = d_prev[j - s_idx]  when j >= s_idx (else NEG_INF)
        NEG_INF_T = cost.new_full((U,), NEG_INF)

        for i in range(1, max_segments + 1):
            d_prev = d[:, i - 1]  # (T+1,)
            padded = torch.cat([NEG_INF_T, d_prev[:T]])  # (U+T,)

            shifted = torch.stack(
                [padded[U - s_idx : U - s_idx + T] for s_idx in range(U)], dim=0
            )  # (U, T)

            candidates = shifted - cost.T

            valid = (shifted > NEG_INF) & (cost.T < float("inf"))
            candidates = torch.where(
                valid, candidates, cost.new_full(candidates.shape, NEG_INF)
            )

            best_vals, best_s_idx = candidates.max(dim=0)
            d[1 : T + 1, i] = best_vals
            parent[1 : T + 1, i] = best_s_idx + 1

        return d, parent

    def _dp_backtrack(
        self,
        parent: torch.Tensor,
        T: int,
        num_segments: int,
    ) -> torch.Tensor:
        """Recover segment_idx from a parent table at state (T, num_segments)."""
        device = parent.device
        seg_lengths: list[int] = []
        j, seg_i = T, num_segments
        while seg_i > 0:
            s = int(parent[j, seg_i].item())
            seg_lengths.append(s)
            j -= s
            seg_i -= 1
        seg_lengths.reverse()

        segment_idx = torch.zeros(T, dtype=torch.long, device=device)
        pos = 0
        for seg_id, length in enumerate(seg_lengths):
            segment_idx[pos : pos + length] = seg_id
            pos += length
        return segment_idx

    def _run_dp(
        self,
        cost: torch.Tensor,
        T: int,
        num_segments: int,
    ) -> torch.Tensor:
        """Run DP and backtrack to recover the optimal segment index tensor.

        Args:
            cost:         (T, max_span) distortion table from
                          ``_compute_cost_table``.
            T:            unpadded sequence length.
            num_segments: target number of segments T' (already clamped for
                          feasibility by the caller).
        Returns:
            segment_idx: (T,) long tensor; 0-indexed segment label per frame.
        """
        d, parent = self._dp_forward(cost, T, num_segments)
        if d[T, num_segments].item() == float("-inf"):
            # Should not happen after feasibility clamping, but guard anyway.
            return self._uniform_segment(T, num_segments, cost.device)
        return self._dp_backtrack(parent, T, num_segments)

    # ------------------------------------------------------------------
    # Anchor-restricted DP — boundaries are a subset of anchor positions
    # ------------------------------------------------------------------

    def _run_dp_anchor_subset(
        self,
        x_b: torch.Tensor,  # (vlen, D)
        anchor_positions: torch.Tensor,  # (K,) sorted positions in [1, vlen)
        num_segments: int,
    ) -> torch.Tensor:
        vlen = x_b.shape[0]
        device = x_b.device

        # Augmented candidate positions: 0, a_1, …, a_K, vlen
        positions = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                anchor_positions,
                torch.tensor([vlen], dtype=torch.long, device=device),
            ]
        )  # (M,) with M = K + 2

        cost_table = self._compute_cost_table(x_b)  # (vlen, max_span)

        # Initialize everything as invalid (infinite cost)
        constrained_cost = torch.full_like(cost_table, float("inf"))

        # ---------------------------------------------------------------
        # FIX 2: Vectorized Span Calculation (No Python Loops)
        # ---------------------------------------------------------------
        # Create a matrix of all possible spans: (M, M)
        p_end = positions.view(-1, 1)  # (M, 1)
        p_start = positions.view(1, -1)  # (1, M)
        spans = p_end - p_start  # (M, M) matrix of all spans

        # Build a boolean mask for valid spans
        mask = (spans > 0) & (spans <= cost_table.shape[1])
        if self.max_span is not None:
            mask &= spans <= self.max_span

        # Extract the valid end positions and valid spans
        valid_ends = p_end.expand_as(spans)[mask]
        valid_spans = spans[mask]

        # Vectorized assignment: copy only valid anchor-to-anchor costs
        constrained_cost[valid_ends - 1, valid_spans - 1] = cost_table[
            valid_ends - 1, valid_spans - 1
        ]

        # Run constrained DP
        seg_idx_b = self._run_dp(constrained_cost, vlen, num_segments)
        return seg_idx_b

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _uniform_segment(T: int, num_segments: int, device) -> torch.Tensor:
        """Evenly distribute T frames into num_segments segments."""
        segment_idx = torch.zeros(T, dtype=torch.long, device=device)
        for t in range(T):
            segment_idx[t] = min(int(t * num_segments / T), num_segments - 1)
        return segment_idx

    # ------------------------------------------------------------------
    # Shared helpers (same pattern as other compression models)
    # ------------------------------------------------------------------

    def _no_rate_output(self, x: torch.Tensor, padding_mask) -> CompressionOutput:
        """Trivial per-frame segmentation returned when rate is None."""
        B, T, _ = x.shape
        device = x.device
        boundary = torch.ones(B, T, device=device, dtype=torch.long)
        boundary[:, 0] = 0
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)
        boundary_soft = boundary.float()
        segment_idx = torch.cumsum(boundary, dim=1)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1
        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=x,
            expected_length=expected_length,
        )

    def _segment_average(
        self,
        x: torch.Tensor,
        segment_idx: torch.Tensor,
        padding_mask,
    ) -> torch.Tensor:
        """Segment-wise mean features, upsampled back to frame resolution.

        Uses scatter_add + gather instead of a dense (B, S, T) assignment
        matrix, reducing peak memory from O(B·S·T·D) to O(B·T·D).
        """
        B, T, D = x.shape
        device = x.device
        max_segments = int(segment_idx.max().item()) + 1

        x_masked = (
            x
            if padding_mask is None
            else x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        )

        # Scatter-add frames into segments → (B, S, D)
        idx_exp = segment_idx.unsqueeze(-1).expand(B, T, D)  # (B, T, D)
        seg_sum = torch.zeros(B, max_segments, D, device=device, dtype=x.dtype)
        seg_sum.scatter_add_(1, idx_exp, x_masked)

        # Count frames per segment (use ones, zero out padding)
        ones = torch.ones(B, T, 1, device=device, dtype=x.dtype)
        if padding_mask is not None:
            ones = ones.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        seg_count = torch.zeros(B, max_segments, 1, device=device, dtype=x.dtype)
        seg_count.scatter_add_(1, segment_idx.unsqueeze(-1), ones)

        seg_means = seg_sum / (seg_count + 1e-6)  # (B, S, D)

        # Gather back to frame resolution → (B, T, D)
        return seg_means.gather(1, idx_exp)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        rate=None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_boundary: Optional[torch.Tensor] = None,
        percent_kept_boundary: Optional[float] = None,
        **kwargs,
    ) -> CompressionOutput:
        """Segment the input sequence using DP-optimal boundary placement.

        Args:
            x:                     (B, T, D) input features.
            rate:                  Compression rate ∈ (0, 1].
                                   Target segments T' = clamp(round(rate × T), 1, T).
                                   Pass ``None`` to return a trivial per-frame
                                   identity segmentation.
            padding_mask:          (B, T) bool tensor; True marks padding positions.
            anchor_boundary:       (B, T) binary tensor of boundary positions
                                   inherited from prior codebooks (1 = boundary).
                                   When provided together with a positive
                                   ``percent_kept_boundary``, **all** anchors are
                                   forced as cuts; the segment count for each
                                   resulting chunk is then jointly optimised by
                                   a 2-D allocation DP, so the result is
                                   globally optimal under the original L2
                                   distortion objective subject to the constraint
                                   "every anchor is a boundary".
            percent_kept_boundary: Switch (kept for API parity with
                                   cosine_similarity).  Any positive value
                                   enables the anchor-augmented path; 0/None
                                   disables it.  Anchors are never subsampled.
                                   When the constraint is infeasible
                                   (num_anchors > num_segments − 1), the code
                                   falls back to plain unconstrained DP.

        Returns:
            CompressionOutput
        """
        B, T, _ = x.shape
        device = x.device

        if rate is None:
            return self._no_rate_output(x, padding_mask)

        if percent_kept_boundary is None:
            percent_kept_boundary = self.percent_kept_boundary
        use_anchors = (
            anchor_boundary is not None
            and percent_kept_boundary is not None
            and percent_kept_boundary > 0
        )

        valid_len = (
            (~padding_mask).sum(dim=1).float() if padding_mask is not None else None
        )

        if isinstance(rate, torch.Tensor):
            rate_b = rate.squeeze(-1)
        else:
            rate_b = torch.full((B,), rate, device=device, dtype=torch.float)

        all_segment_idx = torch.zeros(B, T, dtype=torch.long, device=device)

        for b in range(B):
            vlen = int(valid_len[b].item()) if valid_len is not None else T
            if vlen <= 1:
                continue

            x_b = x[b, :vlen]  # (vlen, D)

            r_t = rate_b[b] if rate_b.dim() > 0 else rate_b  # float32 scalar tensor

            # Replicate CosineSimilarityCompression topk formula exactly,
            # keeping arithmetic in float32 tensors to match its rounding:
            #   no padding   → k = (r × (vlen−1)).long(),       num_segments = k+1
            #   with padding → k = clamp((r×vlen−1).long(), 0), num_segments = k+1
            if valid_len is None:
                num_segments = max(1, int((r_t * (vlen - 1)).long().item()) + 1)
            else:
                num_segments = max(1, int((r_t * vlen - 1).long().item()) + 1)

            # Feasibility: with a finite max_span, we need at least
            # ceil(vlen / max_span) segments to cover all frames.
            if self.max_span is not None:
                num_segments = max(num_segments, math.ceil(vlen / self.max_span))
            num_segments = min(num_segments, vlen)

            # Cost table and DP run without gradient tracking — boundaries are
            # hard decisions; gradients flow through _segment_average instead.
            with torch.no_grad():
                # Anchor-augmented path: select the best subset of anchor
                # positions as boundaries, minimizing total L2 distortion.
                anchors_b = None
                if use_anchors:
                    # Anchors live at positions 1..vlen-1 (position 0 is never a
                    # boundary by convention).
                    anchors_b = anchor_boundary[b, 1:vlen].nonzero(as_tuple=True)[0] + 1
                    if anchors_b.numel() > 0:
                        seg_idx_b = self._run_dp_anchor_subset(
                            x_b,
                            anchors_b,
                            num_segments,
                        )
                    else:
                        cost = self._compute_cost_table(x_b)
                        seg_idx_b = self._run_dp(cost, vlen, num_segments)
                else:
                    cost = self._compute_cost_table(x_b)
                    seg_idx_b = self._run_dp(cost, vlen, num_segments)

            all_segment_idx[b, :vlen] = seg_idx_b
            if vlen < T:
                all_segment_idx[b, vlen:] = int(seg_idx_b[-1].item())

        # Derive boundary tensor from segment indices.
        boundary = torch.zeros(B, T, device=device, dtype=torch.long)
        boundary[:, 1:] = (all_segment_idx[:, 1:] != all_segment_idx[:, :-1]).long()
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)

        boundary_soft = boundary.float()
        # Recompute cumsum-based segment_idx for pipeline consistency.
        segment_idx = torch.cumsum(boundary, dim=1)
        reconstructed = self._segment_average(x, segment_idx, padding_mask)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1

        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=reconstructed,
            expected_length=expected_length,
        )
