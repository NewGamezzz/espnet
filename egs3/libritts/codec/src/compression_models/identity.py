import torch

from .base import BaseCompressionModel, CompressionOutput


class IdentityCompression(BaseCompressionModel):
    """No-op compression: every input frame becomes its own segment.

    Used as a no-compression baseline when evaluating codec reconstruction
    (the codec sees one code per frame, like standard inference, but the
    pipeline still produces a valid CompressionOutput so downstream
    segment-rate stats and decoding code paths work unchanged).

    Forward kwargs (rate, anchor_boundary, ...) are accepted and ignored so
    this is a drop-in replacement in sweeps; the rate has no effect because
    the output is always per-frame segmentation.
    """

    def forward(
        self,
        x: torch.Tensor,
        rate=None,
        padding_mask=None,
        **kwargs,
    ) -> CompressionOutput:
        return self._no_rate_output(x, padding_mask)
