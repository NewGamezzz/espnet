"""Dual-prompt F5: one CFM subclass that masks the deterministic target span.

``DualPromptCFM.forward`` mirrors ``CFM.forward`` from PR 6515 line for line,
consuming the parent's random draws unconditionally so that a row with the
sentinel ``cond_frames = -1`` is bit-identical to stock F5, and replaces the
random infill span with ``[cond_frames, len)`` for every other row. The DiT,
the noise schedule and the loss are untouched.
"""

from __future__ import annotations

from random import random
from typing import Optional

import torch
import torch.nn.functional as F

from espnet3.systems.tts.f5_tts.cfm import CFM
from espnet3.systems.tts.f5_tts.f5tts import F5TTS
from espnet3.systems.tts.f5_tts.utils import (
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class DualPromptCFM(CFM):
    """CFM whose prediction span is the target region when ``cond_frames`` is set."""

    def prediction_mask(
        self, lens: torch.Tensor, cond_frames: torch.Tensor
    ) -> torch.Tensor:
        """Deterministic span ``[cond_frames, lens)`` per row.

        Args:
            lens: ``[B]`` valid frame counts.
            cond_frames: ``[B]`` first target frame per row (``>= 0``).

        Returns:
            Boolean ``[B, max(lens)]`` mask, True on target frames.

        Raises:
            ValueError: If a row's span would be empty.

        Example:
            >>> cfm.prediction_mask(torch.tensor([5]), torch.tensor([2]))
            tensor([[False, False,  True,  True,  True]])
        """
        seq_len = int(lens.max())
        pos = torch.arange(seq_len, device=lens.device)
        cf = cond_frames.to(lens.device).long()
        if bool((cf >= lens).any()):
            raise ValueError(
                "cond_frames leaves an empty target span for at least one row"
            )
        return (pos[None, :] >= cf[:, None]) & (pos[None, :] < lens[:, None])

    def forward(
        self,
        inp,
        text,
        *,
        lens=None,
        cond_frames: Optional[torch.Tensor] = None,
        noise_scheduler=None,
    ):
        """Flow-matching loss with a deterministic or random prediction span.

        Args:
            inp: Mel ``[B, T, D]`` or waveform ``[B, N]``.
            text: Token ids ``[B, T_text]`` (filler -1) or list of strings.
            lens: ``[B]`` valid frames.
            cond_frames: ``[B]`` first target frame; ``-1`` selects the
                parent's random span for that row.
            noise_scheduler: Unused, kept for signature parity.

        Returns:
            ``(loss, cond, pred)`` as the parent.

        Example:
            >>> loss, _, _ = cfm(mel, text=ids, lens=lens, cond_frames=cf)
        """
        if inp.ndim == 2:
            inp = self.mel_spec(inp).permute(0, 2, 1)
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)
        # Same RNG consumption as the parent so sentinel rows stay bit-identical.
        frac_lengths = (
            torch.zeros((batch,), device=device)
            .float()
            .uniform_(*self.frac_lengths_mask)
        )
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths) & mask
        if cond_frames is not None:
            cf = torch.as_tensor(cond_frames, device=device).long().view(-1)
            det = cf >= 0
            if bool(det.any()):
                det_mask = self.prediction_mask(lens, cf.clamp(min=0))
                rand_span_mask = torch.where(det[:, None], det_mask, rand_span_mask)
        x1 = inp
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        drop_audio_cond = random() < self.audio_drop_prob
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False
        pred = self.transformer(
            x=phi,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
        )
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        return loss.mean(), cond, pred


class DualPromptF5TTS(F5TTS):
    """F5TTS whose CFM is :class:`DualPromptCFM`; batch may carry ``cond_frames``."""

    def __init__(self, *args, **kwargs):
        """Build the stock model, then swap the CFM for the subclass.

        All arguments are :class:`F5TTS`'s. The state dict is unchanged by the
        swap, so stock checkpoints load strictly.

        Example:
            .. code-block:: yaml

                model:
                  _target_: src.model.DualPromptF5TTS
                  token_list: ${token_list}
                  hidden_size: 1024
        """
        super().__init__(*args, **kwargs)
        stock = self.cfm
        self.cfm = DualPromptCFM(
            transformer=stock.transformer,
            sigma=stock.sigma,
            odeint_kwargs=stock.odeint_kwargs,
            audio_drop_prob=stock.audio_drop_prob,
            cond_drop_prob=stock.cond_drop_prob,
            num_channels=stock.num_channels,
            mel_spec_module=stock.mel_spec,
            frac_lengths_mask=stock.frac_lengths_mask,
            vocab_char_map=stock.vocab_char_map,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        cond_frames: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Training step; ``cond_frames`` ``[B, 1]`` or ``[B]`` marks the target start.

        Args:
            text: Token ids ``[B, T_text]`` padded with 0.
            text_lengths: ``[B]``.
            speech: Waveforms ``[B, N]``.
            speech_lengths: ``[B]``.
            cond_frames: First target frame per row; ``None`` or ``-1`` rows
                use stock F5's random span.
            **kwargs: Ignored batch fields.

        Returns:
            ``(loss, {"loss": loss}, None)``.

        Example:
            >>> loss, stats, _ = model(cond_frames=cf, **batch)
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        text = self._remap_text_padding(text, text_lengths)
        cf = None if cond_frames is None else torch.as_tensor(cond_frames).view(-1)
        loss, _cond, _pred = self.cfm(
            feats, text=text, lens=feats_lengths, cond_frames=cf
        )
        return loss, dict(loss=loss.detach()), None
