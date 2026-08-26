"""Compression models for the multi-compression codec wrapper.

Ported from https://github.com/NewGamezzz/Multi-Compression-Audio-Codec
(src/compression_models). Each model segments a (B, T, D) frame sequence
into variable-length segments controlled by a ``rate`` parameter and
returns segment-averaged features upsampled back to frame resolution.
"""

from .base import BaseCompressionModel, CompressionOutput
from .cosine_similarity import CosineSimilarityCompression
from .density_peak import DensityPeakCompression
from .dp_segmentation import DPSegmentationCompression
from .identity import IdentityCompression

_REGISTRY = {
    "cosine_similarity": CosineSimilarityCompression,
    "density_peak": DensityPeakCompression,
    "dp_segmentation": DPSegmentationCompression,
    # No-op: every frame becomes its own segment.  Use this as the
    # "no compression" baseline when evaluating codec reconstruction.
    "none": IdentityCompression,
    "identity": IdentityCompression,
}


def build_compression_model(name: str, **kwargs) -> BaseCompressionModel:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown compression model '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "BaseCompressionModel",
    "CompressionOutput",
    "CosineSimilarityCompression",
    "DensityPeakCompression",
    "DPSegmentationCompression",
    "IdentityCompression",
    "build_compression_model",
]
