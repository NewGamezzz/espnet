import torch

from .base import BaseCompressionModel, CompressionOutput


class IdentityCompression(BaseCompressionModel):
    """No-op compression: every input frame becomes its own segment.

    Used as a no-compression baseline when evaluating codec reconstruction
    (the codec sees one code per frame, like standard inference, but the
    encode_with_segments pipeline still produces valid CompressionOutput
    so downstream segment-rate stats and decoding code paths work unchanged).

    All kwargs (rate, padding_mask, max_span, threshold, anchor_*, ...) are
    accepted and ignored so this is a drop-in replacement in scripts that
    sweep rates or pass anchor settings.  The rate has no effect because
    there's nothing to vary — output is always per-frame segmentation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Accept and ignore any kwargs (e.g. max_span, threshold) so configs
        # and CLI flags written for other models still parse cleanly.

    def forward(
        self,
        x: torch.Tensor,
        rate=None,
        padding_mask=None,
        **kwargs,
    ) -> CompressionOutput:
        B, T, _ = x.shape
        device = x.device

        # Boundary at every frame.  Position 0 is the implicit first-segment
        # start and stays 0 in this representation (matches the convention
        # used by CosineSimilarityCompression._no_rate_output).
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
            # Identity: no segment-averaging; pass features through unchanged.
            reconstructed_features=x,
            expected_length=expected_length,
        )
