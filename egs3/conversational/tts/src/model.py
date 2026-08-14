"""espnet3 model wrapper: packed conversation batch -> (loss, stats, weight).

``ESPnetLightningModule`` calls ``model(**batch)`` with the packed-collator
keys (``counts``, ``speech``, ``speech_lengths``, ``text``, ...).  This
wrapper extracts vocos log-mel per packed row (``VocoderMelSpec``, the same
front-end the pretrained F5TTS_Base was trained against) and delegates to
``MultiBranchCFM``.  Text ids arrive padded with -1 from
``collate_conversations``, which is already F5's filler convention, so no
padding remap is needed (unlike the espnet2 ``F5TTS`` wrapper, which undoes
``CommonCollateFn``'s 0-padding).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from egs3.conversational.tts.src.multibranch_cfm import MultiBranchCFM


class MultiBranchF5(nn.Module):
    """Multi-branch F5-TTS flow-matching model for the espnet3 trainer."""

    def __init__(self, cfm: MultiBranchCFM, feats_extract: nn.Module):
        super().__init__()
        self.cfm = cfm
        self.feats_extract = feats_extract

    @property
    def ctx(self):
        """The BranchContext shared with the injected exchanges."""
        return self.cfm.ctx

    def forward(
        self,
        counts,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor = None,
        cond_frames: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Flow-matching training/validation step on one packed batch.

        Args:
            counts: Per-conversation channel counts, ``sum == R``.
            speech: Packed waveform rows ``[R, T_wav]`` at 24 kHz.
            speech_lengths: ``[R]`` valid samples per row.
            text: Masked-script token ids ``[R, T_text]`` padded with -1.
            text_lengths: ``[R]`` (unused; F5 reads the -1 padding directly).
            cond_frames: ``[B]`` long, -1 sentinel; chunk-task conversations'
                deterministic span boundary (collate_conversations, Task 8).
                Passed through to ``MultiBranchCFM.forward`` verbatim.
        """
        feats, feats_lengths = self.feats_extract(speech, speech_lengths)
        loss, ch_stats, _ = self.cfm(
            feats, text, counts=counts, lens=feats_lengths, cond_frames=cond_frames
        )
        stats = {"loss": loss.detach(), **ch_stats}
        weight = loss.new_tensor(len(counts))  # conversations, not rows
        return loss, stats, weight
