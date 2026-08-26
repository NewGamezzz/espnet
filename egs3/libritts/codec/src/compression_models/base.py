from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CompressionOutput:
    """Return type for all compression model forward passes.

    Shapes (B = batch, T = frames, D = feature dim, S = segments):

    - ``boundary_soft``          (B, T)   — float boundary indicators (0 or 1).
    - ``segment_idx``            (B, T)   — int segment index per frame.
    - ``reconstructed_features`` (B, T, D)— segment-averaged features upsampled
                                            back to frame resolution.
    - ``expected_length``        (B, 1)   — number of segments (boundaries + 1).
    """

    boundary_soft: torch.Tensor
    segment_idx: torch.Tensor
    reconstructed_features: torch.Tensor
    expected_length: torch.Tensor


class BaseCompressionModel(nn.Module, ABC):
    """Abstract base class for all compression models.

    Subclasses must implement ``forward``, which segments an input feature
    sequence and returns a ``CompressionOutput``.  The per-frame identity
    output (``rate=None``) and the segment-averaging math are shared here.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        rate=None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionOutput:
        """Segment the input sequence and return compressed representations.

        Args:
            x: (B, T, D) input features.
            rate: Compression rate or budget (interpretation is model-specific).
                  When ``None``, every frame is its own segment (identity).
            padding_mask: (B, T) bool tensor; ``True`` marks padding positions.

        Returns:
            CompressionOutput
        """

    def _no_rate_output(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> CompressionOutput:
        """Trivial per-frame segmentation (used when rate is None)."""
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
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Segment-wise mean features, upsampled back to frame resolution.

        Uses scatter_add + gather instead of a dense (B, S, T) assignment
        matrix, keeping peak memory at O(B·T·D).

        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        device = x.device
        max_segments = int(segment_idx.max().item()) + 1

        x_masked = (
            x
            if padding_mask is None
            else x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        )

        # Scatter-add frames into segments -> (B, S, D)
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

        # Gather back to frame resolution -> (B, T, D)
        return seg_means.gather(1, idx_exp)
