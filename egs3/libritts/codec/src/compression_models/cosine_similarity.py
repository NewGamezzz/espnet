import torch
import torch.nn.functional as F

from .base import BaseCompressionModel, CompressionOutput


class CosineSimilarityCompression(BaseCompressionModel):
    def __init__(self, threshold=1, mode="topk", percent_kept_boundary=None,
                 max_tokens_per_group=None, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.kwargs = kwargs
        self.mode = mode
        self.percent_kept_boundary = percent_kept_boundary
        # Mirror FlexiCodec's max_tokens_per_group: hard cap on segment
        # length.  When set, any segment that would naturally be longer
        # than this gets split into max_tokens_per_group-sized chunks
        # (with a possible smaller leftover at the end of the natural
        # segment).  Applies to all modes; None disables.
        self.max_tokens_per_group = max_tokens_per_group

        if self.percent_kept_boundary is not None and self.mode != "topk":
            warning_msg = (
                "percent_kept_boundary is provided but mode is not 'topk', it will be ignored. "
                "Please set mode='topk' to use percent_kept_boundary or remove percent_kept_boundary if not needed."
            )
            print(f"WARNING: {warning_msg}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _no_rate_output(self, x, padding_mask):
        """Return trivial per-frame segmentation (used when rate=None)."""
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

    def _compute_cosine_sim(self, x, padding_mask):
        """Compute frame-to-frame cosine similarities, masking padding positions.

        Returns:
            cosine_sim: (B, T-1)
            valid_len:  (B,) float or None if no padding mask
        """
        cosine_sim = F.cosine_similarity(x[:, :-1], x[:, 1:], dim=-1)  # (B, T-1)
        valid_len = None
        if padding_mask is not None:
            valid_len = (~padding_mask).sum(dim=1).float()  # (B,)
            cosine_sim = cosine_sim.masked_fill(padding_mask[:, 1:], 1.0)
        return cosine_sim, valid_len

    def _compute_stop_tokens_topk_or_num_segment(
        self, cosine_sim, rate, valid_len, B, T, device, anchor_boundary, percent_kept_boundary
    ):
        """Shared batch loop for 'topk' and 'num_segment' modes."""
        if self.mode == "topk":
            k = (
                torch.clamp(((rate * valid_len) - 1).long(), min=0)
                if valid_len is not None
                else (rate * torch.ones(B, device=device) * (T - 1)).long()
            )
        else:  # num_segment
            if isinstance(rate, torch.Tensor):
                k = torch.clamp((rate.squeeze() - 1).long(), min=0)
            else:
                k = torch.tensor([max(0, int(rate) - 1)] * B, device=device)

        stop_tokens = torch.zeros_like(cosine_sim, dtype=torch.long)
        for b in range(B):
            k_b = k[b].item() if k.dim() > 0 else k.item()
            if k_b <= 0:
                continue

            sim_b = cosine_sim[b].clone()
            indices_b = []

            if anchor_boundary is not None and percent_kept_boundary is not None:
                anchors_b = anchor_boundary[b, 1:] == 1
                k_anchor = min(int(percent_kept_boundary * k_b), int(anchors_b.int().sum().item()))
                if k_anchor > 0:
                    sim_anchors = sim_b.clone()
                    sim_anchors[~anchors_b] = float("inf")
                    _, picked_anchors = torch.topk(sim_anchors, k_anchor, largest=False)
                    indices_b.append(picked_anchors)
                    sim_b[picked_anchors] = float("inf")
                    k_b -= k_anchor

            if k_b > 0:
                _, picked_remaining = torch.topk(sim_b, k_b, largest=False)
                indices_b.append(picked_remaining)

            if indices_b:
                stop_tokens[b, torch.cat(indices_b)] = 1

        return stop_tokens

    def _compute_stop_tokens(self, cosine_sim, rate, valid_len, B, T, device, anchor_boundary, percent_kept_boundary):
        """Dispatch to mode-specific stop-token computation."""
        if self.mode in ["topk", "num_segment"]:
            return self._compute_stop_tokens_topk_or_num_segment(
                cosine_sim, rate, valid_len, B, T, device, anchor_boundary, percent_kept_boundary
            )
        if self.mode == "threshold":
            return (cosine_sim < rate).long()
        if self.mode == "flexicodec":
            # FlexiCodec's threshold-based segmentation
            # (modeling_flexicodec.py::_perform_similarity_alignment_vectorized).
            # A new segment starts wherever consecutive-frame cosine similarity
            # is AT OR BELOW ``rate`` -- i.e. the two frames are dissimilar
            # enough that the latter is judged to belong to a new segment.
            # Fully vectorised; the "iterate left-to-right" framing in some
            # descriptions is just informal -- the underlying op is a single
            # element-wise comparison.  Differs from the existing "threshold"
            # mode only in the inequality (<= vs <), which matters at the
            # exact-equality edge case.
            return (cosine_sim <= rate).long()
        raise ValueError(f"Invalid mode {self.mode}. Choose from 'topk', 'threshold', 'num_segment', or 'flexicodec'.")

    def _build_boundary(self, stop_tokens, B, device, padding_mask):
        """Prepend t=0 zero and apply padding mask to produce a (B, T) boundary tensor."""
        boundary = torch.cat(
            [torch.zeros(B, 1, device=device, dtype=stop_tokens.dtype), stop_tokens],
            dim=1,
        )
        if padding_mask is not None:
            boundary = boundary.masked_fill(padding_mask, 0)
        return boundary

    def _apply_max_segment_length(self, boundary, padding_mask=None):
        """Force a boundary at every position where the current segment would
        otherwise exceed ``self.max_tokens_per_group`` frames.  No-op when
        ``self.max_tokens_per_group`` is None.

        Mirrors FlexiCodec's vectorised cummax trick: compute the in-segment
        index of every frame, then add a boundary wherever this index is a
        positive multiple of the max.  Combined (OR'd) with the existing
        natural boundaries; original natural splits are preserved exactly.

        Args:
            boundary: (B, T) int tensor; 1 marks the START of a new segment
                      (pos 0 is the implicit first-segment start and stays 0
                      in this representation).
            padding_mask: (B, T) optional; True = padding.  Forced boundaries
                          falling on padded positions are cleared.

        Returns:
            new boundary tensor with the same shape and dtype.
        """
        if self.max_tokens_per_group is None:
            return boundary

        B, T = boundary.shape
        device = boundary.device

        # FlexiCodec convention: treat position 0 as an implicit segment
        # start when computing the in-segment index.  We don't modify the
        # output boundary's pos 0 (which stays 0 by our convention).
        is_seg_start = boundary.clone().long()
        is_seg_start[:, 0] = 1

        arange_t = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)
        segment_start_markers = arange_t * is_seg_start
        last_seg_start = torch.cummax(segment_start_markers, dim=1).values
        in_seg_idx = arange_t - last_seg_start

        # Force a boundary at every n*max_tokens position WITHIN a segment.
        # Skip in_seg_idx == 0 (already a boundary or implicit start).
        forced = (in_seg_idx > 0) & ((in_seg_idx % self.max_tokens_per_group) == 0)
        new_boundary = (boundary.bool() | forced).to(boundary.dtype)

        if padding_mask is not None:
            new_boundary = new_boundary.masked_fill(padding_mask, 0)
        return new_boundary

    def _segment_average(self, x, segment_idx, padding_mask):
        """Compute segment-wise average features and upsample back to frame level.

        Returns:
            average_vectors: (B, T, D)
        """
        device = x.device
        max_segments = segment_idx.max().item() + 1
        seg_ids = torch.arange(max_segments, device=device)[None, :, None]  # (1, S, 1)
        assign = (segment_idx.unsqueeze(1) == seg_ids).float()              # (B, S, T)
        if padding_mask is not None:
            assign = assign.masked_fill(padding_mask.unsqueeze(1), 0.0)
        assign_norm = assign / (assign.sum(dim=2, keepdim=True) + 1e-6)
        segment_means = torch.matmul(assign_norm, x)                        # (B, S, D)
        return torch.matmul(assign.transpose(1, 2), segment_means)          # (B, T, D)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x, rate=None, padding_mask=None, anchor_boundary=None, percent_kept_boundary=None, **kwargs) -> CompressionOutput:
        """
        Args:
            x: (B, T, D)
            padding_mask: (B, T), True = padding
            anchor_boundary: (B, T) binary tensor indicating anchor boundaries to include

        Returns:
            CompressionOutput
        """
        B, T, _ = x.shape
        device = x.device

        if percent_kept_boundary is None:
            percent_kept_boundary = self.percent_kept_boundary

        if rate is None:
            return self._no_rate_output(x, padding_mask)

        cosine_sim, valid_len = self._compute_cosine_sim(x, padding_mask)
        stop_tokens = self._compute_stop_tokens(
            cosine_sim, rate, valid_len, B, T, device, anchor_boundary, percent_kept_boundary
        )
        boundary = self._build_boundary(stop_tokens, B, device, padding_mask)
        # Optional FlexiCodec-style hard cap on segment length.  No-op when
        # ``self.max_tokens_per_group is None``.
        boundary = self._apply_max_segment_length(boundary, padding_mask)
        boundary_soft = boundary.float()
        segment_idx = torch.cumsum(boundary, dim=1)
        average_vectors = self._segment_average(x, segment_idx, padding_mask)
        expected_length = boundary_soft.sum(dim=1, keepdim=True) + 1

        return CompressionOutput(
            boundary_soft=boundary_soft,
            segment_idx=segment_idx,
            reconstructed_features=average_vectors,
            expected_length=expected_length,
        )

