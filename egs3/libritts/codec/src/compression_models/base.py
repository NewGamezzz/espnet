import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


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
    sequence and returns a ``CompressionOutput``.

    ``predict_segments`` is provided as a concrete no-grad wrapper and does
    not need to be overridden.
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

    @torch.no_grad()
    def predict_segments(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionOutput:
        """No-grad wrapper around ``forward`` for inference-time use."""
        return self.forward(x, padding_mask=padding_mask, **kwargs)
