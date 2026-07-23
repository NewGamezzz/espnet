"""Multi-compression RVQ wrappers.

Two layers of wrapping:

- ``RVQCompressionWrapper`` wraps espnet2's ``ResidualVectorQuantization``
  (``espnet2/gan_codec/shared/quantizer/modules/core_vq.py``) and inserts a
  compression-model call before every codebook layer.  Its forward/encode/
  decode mirror the original loops exactly, so with the identity compression
  model (or ``rate=None``) it is bit-equivalent to the unwrapped module.
- ``CompressionResidualVectorQuantizer`` replaces espnet2's
  ``ResidualVectorQuantizer`` (``generator.quantizer``) while keeping its
  exact call interface: ``forward(x, sample_rate, bandwidth)``,
  ``encode(x, sample_rate, bandwidth, st)``, ``decode(codes)``.  Because
  every ``espnet2/gan_codec`` RVQ codec (SoundStream, Encodec, DAC,
  FunCodec) calls its quantizer through exactly this interface, swapping
  the module is all that is needed — the codec's own forward/encode/decode
  and losses run unmodified.

Compression-rate delivery (the codecs never pass a ``rate`` kwarg):

- training: sampled internally per forward pass (``random_rate`` strategy in
  [``min_rate``, ``max_rate``]).  SoundStream/Encodec cache generator
  outputs across the G/D turns of one batch during training, so a single
  sample per batch is used consistently by both turns.
- validation: deterministic ``eval_rate`` (no generator caching in eval,
  and comparable epochs need a fixed rate).
- inference: ``set_inference_rate(rate)`` / ``reset_inference_rate()``,
  because ``codec.encode()`` does not forward extra kwargs.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

RATE_STRATEGIES = ("per_quantizer", "per_sample", "none")


class RVQCompressionWrapper(nn.Module):
    """Multi-mode compression wrapper around ``ResidualVectorQuantization``.

    Applies the compression model to the residual before every codebook
    layer, so each layer quantizes segment-averaged features under its own
    (possibly different) compression rate.
    """

    def __init__(self, original_vq, compression_model):
        super().__init__()
        self.rvq = original_vq  # espnet2 ResidualVectorQuantization
        self.compression_model = compression_model

    @property
    def layers(self):
        """Original codebook layers (property, not an attribute, so the
        modules are not registered twice in the state dict)."""
        return self.rvq.layers

    @property
    def quantizer_dropout(self):
        return getattr(self.rvq, "quantizer_dropout", 0.0)

    def forward(
        self,
        x,
        rate=None,
        n_q: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ):
        quantized_out = 0.0
        residual = x
        rate_layer = rate
        anchor_boundary = None

        if not self.quantizer_dropout:
            all_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)

            for idx, layer in enumerate(self.layers[:n_q]):
                if (
                    rate is not None
                    and isinstance(rate, torch.Tensor)
                    and rate.dim() == 2
                ):
                    rate_layer = rate[idx, :]

                use_anchor = (
                    anchor_start_layer is not None and idx >= anchor_start_layer
                )
                compression_output = self.compression_model(
                    residual.transpose(1, 2),
                    rate=rate_layer,
                    padding_mask=padding_mask,
                    anchor_boundary=anchor_boundary if use_anchor else None,
                )
                compressed_residual = (
                    compression_output.reconstructed_features.transpose(1, 2)
                )

                # Accumulate the boundary union from free layers only; it
                # becomes the anchor set for layers >= anchor_start_layer.
                if anchor_start_layer is not None and not use_anchor:
                    anchor_boundary = (
                        torch.logical_or(
                            anchor_boundary, compression_output.boundary_soft.bool()
                        )
                        if anchor_boundary is not None
                        else compression_output.boundary_soft.bool()
                    )

                quantized, indices, loss = layer(compressed_residual)
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)
                all_losses.append(loss)

            if self.training:
                # Solving subtle bug with STE and RVQ
                # For more, https://github.com/facebookresearch/encodec/issues/25
                quantized_out = x + (quantized_out - x).detach()

            out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
            return quantized_out, out_indices, out_losses
        else:
            all_commit_losses = []
            all_quant_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)
            if self.training:
                n_q = torch.ones((x.shape[0],)) * len(self.layers) + 1
                dropout = torch.randint(1, len(self.layers) + 1, (x.shape[0],))
                n_dropout = int(x.shape[0] * self.quantizer_dropout)
                n_q[:n_dropout] = dropout[:n_dropout]
                n_q = n_q.to(x.device)

            for i, layer in enumerate(self.layers):
                if self.training is False and i >= n_q:
                    break
                mask = torch.full((x.shape[0],), fill_value=i, device=x.device) < n_q

                if (
                    rate is not None
                    and isinstance(rate, torch.Tensor)
                    and rate.dim() == 2
                ):
                    rate_layer = rate[i, :]

                use_anchor = anchor_start_layer is not None and i >= anchor_start_layer
                compression_output = self.compression_model(
                    residual.transpose(1, 2),
                    rate=rate_layer,
                    padding_mask=padding_mask,
                    anchor_boundary=anchor_boundary if use_anchor else None,
                )
                compressed_residual = (
                    compression_output.reconstructed_features.transpose(1, 2)
                )

                if anchor_start_layer is not None and not use_anchor:
                    anchor_boundary = (
                        torch.logical_or(
                            anchor_boundary, compression_output.boundary_soft.bool()
                        )
                        if anchor_boundary is not None
                        else compression_output.boundary_soft.bool()
                    )

                quantized, indices, commit_loss, quant_loss = layer(
                    compressed_residual, mask
                )
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized * mask[:, None, None]

                all_indices.append(indices)
                all_commit_losses.append(commit_loss)
                all_quant_losses.append(quant_loss)

            if self.training:
                quantized_out = x + (quantized_out - x).detach()

            out_commit_losses, out_quant_losses, out_indices = map(
                torch.stack, (all_commit_losses, all_quant_losses, all_indices)
            )
            return quantized_out, out_indices, out_commit_losses, out_quant_losses

    def _encode_loop(
        self,
        x: torch.Tensor,
        rate=None,
        n_q: Optional[int] = None,
        st: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Shared encode loop used by both encode() and encode_with_segments().

        Returns:
            out_indices:   (n_q, B, T') stacked code indices.
            out_quantized: (n_q, B, D, T') pre-quantization residuals per layer.
            segment_indices: list of (B, T') segment index tensors per layer.
        """
        residual = x
        all_indices = []
        all_quantized = []
        segment_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        rate_layer = rate
        anchor_boundary = None

        for idx, layer in enumerate(self.layers[st:n_q]):
            if rate is not None and isinstance(rate, torch.Tensor) and rate.dim() == 2:
                rate_layer = rate[st + idx, :]  # (Q, B) -> (B,)

            abs_idx = st + idx
            use_anchor = (
                anchor_start_layer is not None and abs_idx >= anchor_start_layer
            )
            all_quantized.append(residual)
            compression_output = self.compression_model(
                residual.transpose(1, 2),
                rate=rate_layer,
                padding_mask=padding_mask,
                anchor_boundary=anchor_boundary if use_anchor else None,
            )
            compressed_residual = compression_output.reconstructed_features.transpose(
                1, 2
            )
            segment_indices.append(compression_output.segment_idx)

            if anchor_start_layer is not None and not use_anchor:
                anchor_boundary = (
                    torch.logical_or(
                        anchor_boundary, compression_output.boundary_soft.bool()
                    )
                    if anchor_boundary is not None
                    else compression_output.boundary_soft.bool()
                )

            indices = layer.encode(compressed_residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)

        return torch.stack(all_indices), torch.stack(all_quantized), segment_indices

    def encode(
        self,
        x: torch.Tensor,
        rate=None,
        n_q: Optional[int] = None,
        st: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode to discrete codes. Returns (n_q, B, T') code indices."""
        out_indices, _, _ = self._encode_loop(
            x,
            rate=rate,
            n_q=n_q,
            st=st,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
        )
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out

    def encode_with_segments(
        self,
        x: torch.Tensor,
        rate=None,
        n_q: Optional[int] = None,
        st: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode and also return pre-quantization residuals and segment indices."""
        return self._encode_loop(
            x,
            rate=rate,
            n_q=n_q,
            st=st,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
        )


class CompressionResidualVectorQuantizer(nn.Module):
    """Drop-in replacement for espnet2's ``ResidualVectorQuantizer``.

    Holds the original quantizer as ``self.rvq`` and swaps its inner
    ``.vq`` for an ``RVQCompressionWrapper``.  The public interface
    (``forward(x, sample_rate, bandwidth)`` / ``encode`` / ``decode`` and
    the bandwidth getters) matches the original exactly, including the
    ``quantizer_dropout``-dependent 4- vs 5-tuple forward return.

    Args:
        original_rvq: espnet2 ``ResidualVectorQuantizer`` instance
            (``generator.quantizer`` of a built codec).
        compression_model: a ``BaseCompressionModel``.
        min_rate / max_rate: sampling range for training-time rates.
        random_rate: sampling strategy — ``per_quantizer`` (one rate per
            codebook layer and batch item), ``per_sample`` (one per batch
            item), or ``none`` (no compression during training).
        eval_rate: deterministic rate used in eval-mode forward passes
            (validation).  ``None`` means no compression at validation.
    """

    def __init__(
        self,
        original_rvq,
        compression_model,
        min_rate: float = 0.2,
        max_rate: float = 1.0,
        random_rate: str = "per_quantizer",
        eval_rate: Optional[float] = None,
    ):
        super().__init__()
        if random_rate not in RATE_STRATEGIES:
            raise ValueError(
                f"Unknown random_rate strategy '{random_rate}'. "
                f"Choose from: {RATE_STRATEGIES}."
            )
        if not hasattr(original_rvq, "vq"):
            raise TypeError(
                "CompressionResidualVectorQuantizer expects a quantizer with an "
                "inner '.vq' ResidualVectorQuantization module (espnet2 "
                f"ResidualVectorQuantizer); got {type(original_rvq).__name__}. "
                "HiFiCodec's GroupResidualVectorQuantization is not supported."
            )
        self.rvq = original_rvq
        self.rvq.vq = RVQCompressionWrapper(self.rvq.vq, compression_model)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.random_rate = random_rate
        self.eval_rate = eval_rate
        self._inference_rate = None
        self._inference_anchor_start_layer = None
        # Accept unwrapped-layout checkpoints transparently: a pretrained
        # ResidualVectorQuantizer state dict ("vq.layers...") is remapped to
        # the wrapped layout ("rvq.vq.rvq.layers...") at load time, so a
        # baseline checkpoint can be loaded strictly into the wrapped model
        # even after construction (e.g. rate-sweeping baseline weights at
        # inference without fine-tuning).
        # (_register_* is the stable-across-versions spelling of the public
        # register_load_state_dict_pre_hook.)
        self._register_load_state_dict_pre_hook(self._remap_unwrapped_state_dict)

    def _remap_unwrapped_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Remap original-quantizer keys to the wrapped layout in place.

        Wrapped-layout keys (``rvq.*``) are left untouched, so fine-tuned
        checkpoints load unchanged; when both layouts are present (a merged
        dict), the unwrapped entries win.
        """
        old_prefix = f"{prefix}vq."
        new_prefix = f"{prefix}rvq.vq.rvq."
        for key in [k for k in state_dict if k.startswith(old_prefix)]:
            state_dict[new_prefix + key[len(old_prefix) :]] = state_dict.pop(key)

    # ------------------------------------------------------------------
    # Rate handling
    # ------------------------------------------------------------------

    @property
    def compression_model(self):
        return self.rvq.vq.compression_model

    @property
    def num_codebooks(self) -> int:
        return len(self.rvq.vq.layers)

    def set_inference_rate(self, rate) -> None:
        """Set the compression rate used when no explicit rate is passed.

        Needed at inference because ``codec.encode()`` does not forward
        extra kwargs down to the quantizer.
        """
        self._inference_rate = rate

    def reset_inference_rate(self) -> None:
        self._inference_rate = None

    def set_inference_anchor_start_layer(self, layer: Optional[int]) -> None:
        """Set the anchor start layer used by encode when none is passed.

        Layers at/after this index constrain their boundaries to the union
        of the earlier (free) layers' boundaries.  Inference-only: like the
        inference rate, it exists because ``codec.encode()`` forwards no
        extra kwargs.  Reset after use so training is never affected.
        """
        self._inference_anchor_start_layer = layer

    def reset_inference_anchor_start_layer(self) -> None:
        self._inference_anchor_start_layer = None

    def _sample_rate(self, batch_size: int, device) -> Optional[torch.Tensor]:
        """Sample a training-time rate tensor according to ``random_rate``.

        Ported from the source repo's ``BaseCodecLightningModule._sample_rate``
        (generator-side variant; one sample per batch serves both GAN turns
        because SoundStream/Encodec cache generator outputs in training).
        """
        n_q = self.num_codebooks
        span = self.max_rate - self.min_rate

        if self.random_rate == "per_quantizer":
            return torch.rand(n_q, batch_size, device=device) * span + self.min_rate

        if self.random_rate == "per_sample":
            return torch.rand(batch_size, device=device) * span + self.min_rate

        # "none": no compression during training.
        return None

    def _resolve_rate(self, batch_size: int, device, rate):
        if rate is not None:
            return rate
        if self.training:
            return self._sample_rate(batch_size, device)
        if self._inference_rate is not None:
            return self._inference_rate
        return self.eval_rate

    # ------------------------------------------------------------------
    # ResidualVectorQuantizer interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        rate=None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
    ):
        """Residual vector quantization with per-layer compression.

        Mirrors ``ResidualVectorQuantizer.forward(x, sample_rate,
        bandwidth)``; the extra kwargs are only used when calling the
        wrapper directly (the codecs themselves never pass them).
        """
        rate = self._resolve_rate(x.size(0), x.device, rate)
        bw_per_q = self.rvq.get_bandwidth_per_quantizer(sample_rate)
        n_q = self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)

        if not self.rvq.quantizer_dropout:
            quantized, codes, commit_loss = self.rvq.vq(
                x,
                n_q=n_q,
                rate=rate,
                padding_mask=padding_mask,
                anchor_start_layer=anchor_start_layer,
            )
            bw = torch.tensor(n_q * bw_per_q).to(x)
            return quantized, codes, bw, torch.mean(commit_loss)
        else:
            quantized, codes, commit_loss, quantization_loss = self.rvq.vq(
                x,
                n_q=n_q,
                rate=rate,
                padding_mask=padding_mask,
                anchor_start_layer=anchor_start_layer,
            )
            bw = torch.tensor(n_q * bw_per_q).to(x)
            return (
                quantized,
                codes,
                bw,
                torch.mean(commit_loss),
                torch.mean(quantization_loss),
            )

    def encode(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        st: Optional[int] = None,
        rate=None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode to codes at the given bandwidth; (n_q, B, T') indices."""
        rate = self._resolve_rate(x.size(0), x.device, rate)
        if anchor_start_layer is None:
            anchor_start_layer = self._inference_anchor_start_layer
        n_q = self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        return self.rvq.vq.encode(
            x,
            n_q=n_q,
            st=st or 0,
            rate=rate,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
            **kwargs,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        return self.rvq.vq.decode(codes)

    def encode_with_segments(
        self,
        x: torch.Tensor,
        sample_rate: int,
        bandwidth: Optional[float] = None,
        st: Optional[int] = None,
        rate=None,
        padding_mask: Optional[torch.Tensor] = None,
        anchor_start_layer: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode and also return per-layer residuals and segment indices."""
        rate = self._resolve_rate(x.size(0), x.device, rate)
        if anchor_start_layer is None:
            anchor_start_layer = self._inference_anchor_start_layer
        n_q = self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        return self.rvq.vq.encode_with_segments(
            x,
            n_q=n_q,
            st=st or 0,
            rate=rate,
            padding_mask=padding_mask,
            anchor_start_layer=anchor_start_layer,
            **kwargs,
        )

    def get_num_quantizers_for_bandwidth(
        self, sample_rate: int, bandwidth: Optional[float] = None
    ) -> int:
        return self.rvq.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        return self.rvq.get_bandwidth_per_quantizer(sample_rate)
