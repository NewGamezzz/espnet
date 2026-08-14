"""Multi-branch conditional flow matching on packed conversation batches.

``MultiBranchCFM`` fine-tunes one shared F5-TTS backbone on N parallel
channels of a conversation window: every channel is a row of the packed
batch, and the injected ``branch_exchange`` modules (active inside
``ctx.branches(counts)``) are the only cross-channel communication.
"""

from __future__ import annotations

from random import random

import torch
import torch.nn.functional as F

from egs3.conversational.tts.src.branch_exchange import BranchContext
from espnet2.tts.f5.cfm import CFM
from espnet2.tts.f5.utils import lens_to_mask, mask_from_frac_lengths


def conversation_offsets(counts: torch.Tensor) -> torch.Tensor:
    """Row index of each conversation's first branch: exclusive cumsum of counts."""
    return torch.cumsum(counts, dim=0) - counts


class MultiBranchCFM(CFM):
    """``CFM`` whose ``forward`` trains N branches of a conversation jointly.

    Copy-adapt of ``CFM.forward`` (espnet2/tts/f5/cfm.py) with exactly these
    changes:

    1. Inputs arrive packed (step-2 collator): row-stacked mel ``(R, T, d)``
       and text ``(R, T_text)`` with ``R = sum(counts)``, plus the
       per-conversation ``counts`` list and per-row ``lens (R,)`` (channels
       of one window share one length).  Rows are already transformer-ready;
       no fold is needed.
    2. Shared span: ``frac_lengths`` is sampled per conversation ``(B,)``,
       the span mask is built once per conversation and
       ``repeat_interleave``-d with ``counts`` to ``(R, T)``.  The infilling
       region is time-aligned across all N channels: the model generates
       everyone jointly while every channel's unmasked remainder acts as
       that speaker's voice prompt.
    3. Shared flow time: ``time`` is sampled per conversation ``(B,)`` and
       ``repeat_interleave``-d to ``(R,)``.  At inference all channels ride
       one ODE trajectory at a common t; training must match.  Noise ``x0``
       stays independent per channel.
    4. CFG drops are unchanged (already whole-batch Python scalars).
    5. The transformer runs inside ``with self.ctx.branches(counts)`` so the
       injected exchanges are active.
    6. Loss is the same masked MSE; per-channel means (``loss_ch0``, ...)
       are additionally returned in a stats dict for logging.
    7. For testability, ``forward`` accepts optional pre-sampled
       ``frac_lengths``, ``time`` and ``x0`` (``None`` = sample as usual)
       and returns ``(loss, stats, extras)`` instead of CFM's
       ``(loss, cond, pred)``; ``extras`` carries ``cond``/``pred`` plus the
       sampled quantities.
    """

    def __init__(self, transformer, ctx: BranchContext | None = None, **kwargs):
        super().__init__(transformer=transformer, **kwargs)
        # Plain attribute: BranchContext is deliberately not an nn.Module, so
        # it never appears in the state dict.  It must be the SAME object the
        # exchanges were injected with (see build_model.build_multibranch_f5).
        self.ctx = ctx if ctx is not None else BranchContext()

    def forward(
        self,
        inp,  # (R, T, d) mel or (R, T_wav) raw wave, packed rows
        text,  # (R, T_text) token ids padded with -1
        *,
        counts,  # per-conversation branch counts, sum == R
        lens=None,  # (R,) valid mel frames per row
        noise_scheduler: str | None = None,
        frac_lengths=None,  # (B,) pre-sampled span fractions (tests only)
        time=None,  # (B,) pre-sampled flow times (tests only)
        x0=None,  # (R, T, d) pre-sampled noise (tests only)
        cond_frames=None,  # (B,) long, -1 sentinel: chunk-task deterministic span
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        rows, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        counts_t = torch.as_tensor(counts, dtype=torch.long, device=device)
        n_conv = int(counts_t.numel())
        if int(counts_t.sum()) != rows:
            raise ValueError(f"sum(counts)={int(counts_t.sum())} != packed rows {rows}")

        # lens and mask (per packed row)
        if lens is None:
            lens = torch.full((rows,), seq_len, device=device, dtype=torch.long)
        mask = lens_to_mask(lens, length=seq_len)

        # Channels of one window share one length; the shared span mask
        # below is only well-defined under that invariant, so fail loudly.
        offsets = conversation_offsets(counts_t)
        conv_lens = lens[offsets]
        if not torch.equal(conv_lens.repeat_interleave(counts_t), lens):
            raise ValueError("rows of one conversation must share one length")

        # get a random span to mask out for training conditionally:
        # sampled ONCE per conversation, expanded to its rows.
        if frac_lengths is None:
            frac_lengths = (
                torch.zeros((n_conv,), device=device)
                .float()
                .uniform_(*self.frac_lengths_mask)
            )
        conv_span_mask = mask_from_frac_lengths(conv_lens, frac_lengths)
        conv_span_mask = F.pad(
            conv_span_mask, (0, seq_len - conv_span_mask.shape[1]), value=False
        )

        # Chunk-task conversations (cond_frames >= 0) override the random
        # span with the deterministic target region [cond_frames, conv_len):
        # everything before cond_frames is prompt/prev-chunk context, never
        # masked.  Sentinel rows (-1) fall through to the random draw above
        # untouched, so an all-sentinel batch is bit-identical to omitting
        # the kwarg (the random draw itself stays unconditional).
        if cond_frames is not None:
            cond_frames = torch.as_tensor(cond_frames, device=device, dtype=torch.long)
            if cond_frames.numel() != n_conv:
                raise ValueError(
                    f"cond_frames has {cond_frames.numel()} entries for "
                    f"{n_conv} conversations"
                )
            det = cond_frames >= 0
            if det.any():
                pos = torch.arange(seq_len, device=device)
                det_mask = (pos[None, :] >= cond_frames.clamp(min=0)[:, None]) & (
                    pos[None, :] < conv_lens[:, None]
                )
                if (det & ~det_mask.any(dim=1)).any():
                    raise ValueError(
                        "cond_frames leaves an empty deterministic span for at "
                        "least one conversation (cond_frames >= conv_len)"
                    )
                conv_span_mask = torch.where(det[:, None], det_mask, conv_span_mask)

        rand_span_mask = conv_span_mask.repeat_interleave(counts_t, dim=0)
        rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise, independent per channel
        if x0 is None:
            x0 = torch.randn_like(x1)

        # time step: shared per conversation, expanded to its rows
        if time is None:
            time = torch.rand((n_conv,), dtype=dtype, device=device)
        time_rows = time.repeat_interleave(counts_t)

        # sample xt (phi_t(x) in the paper)
        t = time_rows.unsqueeze(-1).unsqueeze(-1)
        xt = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        with self.ctx.branches(counts, device=device):
            pred = self.transformer(
                x=xt,
                cond=cond,
                text=text,
                time=time_rows,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                mask=mask,
            )

        # flow matching loss
        loss_full = F.mse_loss(pred, flow, reduction="none")
        loss = loss_full[rand_span_mask].mean()

        # Per-channel means: channel k = the k-th row of each conversation
        # (the collator packs rows in branch order).
        row_pos = torch.arange(rows, device=device) - offsets.repeat_interleave(
            counts_t
        )
        stats = {}
        for k in range(int(counts_t.max())):
            sel = row_pos == k
            stats[f"loss_ch{k}"] = loss_full[sel][rand_span_mask[sel]].mean().detach()

        extras = {
            "cond": cond,
            "pred": pred,
            "rand_span_mask": rand_span_mask,
            # frac_lengths is the random draw (discarded if cond_frames >= 0)
            "frac_lengths": frac_lengths,
            "time": time,
        }
        return loss, stats, extras

    def sample(self, cond, text, duration, *, counts, seed=None, **kwargs):
        """``CFM.sample`` with the exchanges active.

        ``cond``/``text`` rows are the packed channels described by
        ``counts`` (for one window: ``counts=[N]``).  All rows ride one ODE
        trajectory at a common t, matching the shared-time training;
        CFG's cond/uncond concatenation is handled by the context's
        segment-aware conversation ids.

        ``seed`` is intercepted: ``CFM.sample`` re-seeds the RNG before
        EVERY row's noise draw (upstream batch-size invariance), which here
        would give all channels of a window bit-identical y0 - off the
        training distribution, where noise is independent per channel.
        Seeding once and letting rows draw sequentially keeps runs
        reproducible AND channels independent (reproducibility becomes
        conditional on the window's N/duration, fine for sanity generation).
        """
        if seed is not None:
            torch.manual_seed(seed)
        with self.ctx.branches(counts, device=cond.device):
            return super().sample(cond, text, duration, seed=None, **kwargs)
