"""Per-row audio-offset extraction for a collated SpeechLM batch (Task 6).

``compute_audio_offsets`` is what step-3 training hands to the branch-exchange
context's ``ctx.branches(align_offsets=...)`` (see
``src/branch_exchange/inject.py``): TAC must be exactly identity on every
pre-audio position, and ``ctx.branches`` needs, per row, the index where that
row's audio region starts.

IMPORTANT (Task 6 finding, see ``tests/test_preprocessing_parity.py``'s
``TestRealPreprocessorParity`` for the reproduction): on BagPiper's real
``SpeechLMPreprocessor.collate_fn`` output, the "stream-0 token in a
``discrete_audio`` vocab interval" test below correctly finds NOTHING and
raises ``ValueError`` on every real training row. That is not a bug in this
module - it reflects how the pipeline actually works.
``espnet2/speechlm/model/speechlm/lm/parallel.py``'s ``ParallelHFModel._embed``
encodes the raw waveform (``discrete_audio_feats``) and writes the resulting
codec ids into ``seqs`` IN PLACE, keyed by ``discrete_audio_indices``, only
during the model forward pass; before that, the audio region of a freshly
collated ``seqs`` is the ``<|pad|>`` (id 0) placeholder, indistinguishable
from ordinary padding. So this helper's stream-0 scan only ever finds a real
audio offset on a batch whose audio region already carries real per-stream
codec ids - e.g. hand-built tensors (see the local tests), or a ``seqs``
tensor captured AFTER ``_embed`` has run. It is NOT usable directly on the
raw ``collate_fn`` output. For that case, the correct pre-forward offset is
``batch["discrete_audio_indices"][:, 1]`` (the preprocessor's own structural
start-position record - see ``speechlm_job.py``'s ``preprocessing`` step 3.4);
step-3 training must source ``align_offsets`` from there for real batches,
not from this helper. This module still implements the literally-specified
stream-0 scan (rather than silently switching to ``discrete_audio_indices``)
because it is a legitimate, simpler, self-contained utility for any batch
that DOES carry real per-stream codes, and switching its semantics without
flagging would hide the real finding instead of surfacing it.
"""

from __future__ import annotations

import torch


def compute_audio_offsets(seqs: torch.Tensor, vocab_intervals: dict) -> torch.LongTensor:
    """Per-row index of the FIRST position whose stream-0 token falls in any
    ``vocab_intervals["discrete_audio"]`` interval.

    Args:
        seqs: Collated token sequence, shape ``(B, T, n_stream)`` (the
            ``SpeechLMPreprocessor.collate_fn`` layout - stream 0 is column 0
            of the last dim) or ``(B, T)`` (already stream-0-only). Any other
            number of dimensions raises ``ValueError``.
        vocab_intervals: The job/model's vocabulary interval map (see
            ``SpeechLMJobTemplate.vocab_intervals`` / the ``tiny_parallel_llm``
            fixture's ``model.vocab_intervals``); ``vocab_intervals["discrete_audio"]``
            is a list of ``(start, end)`` tuples, one per audio stream. A
            stream-0 token counts as "audio" if it falls in ANY of these
            intervals, not just the first (delay-interleaved streams can
            place any stream's codebook in column 0 depending on layout).

    Returns:
        ``torch.LongTensor`` of shape ``(B,)``, one offset per row, on the
        same device as ``seqs``.

    Raises:
        ValueError: ``seqs`` is not 2-D or 3-D, or any row has no stream-0
            token in a ``discrete_audio`` interval (a training row must have
            audio; silently returning a sentinel would hide a real data bug).
    """
    if seqs.dim() == 3:
        stream0 = seqs[:, :, 0]
    elif seqs.dim() == 2:
        stream0 = seqs
    else:
        raise ValueError(
            f"seqs must have shape (B, T) or (B, T, n_stream), got {tuple(seqs.shape)}"
        )

    intervals = vocab_intervals["discrete_audio"]
    is_audio = torch.zeros_like(stream0, dtype=torch.bool)
    for start, end in intervals:
        is_audio |= (stream0 >= start) & (stream0 < end)

    has_audio = is_audio.any(dim=1)
    if not bool(has_audio.all()):
        missing = (~has_audio).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"row(s) {missing} have no stream-0 token in any discrete_audio "
            "interval; every training row must have audio"
        )

    # First True index per row. Not `is_audio.long().argmax(dim=1)`: argmax's
    # tie-break on equal (0/1) values is an implementation detail, not a
    # documented contract - this computes the minimum matching position
    # directly instead of relying on it.
    t = stream0.shape[1]
    positions = torch.arange(t, device=seqs.device)
    candidates = torch.where(is_audio, positions.unsqueeze(0), t)
    offsets = candidates.min(dim=1).values
    return offsets.long()


def compute_audio_offsets_from_batch(batch: dict, vocab_intervals: dict) -> torch.LongTensor:
    """Convenience wrapper unwrapping a ``collate_fn`` output dict.

    ``batch["seqs"]`` is the key ``SpeechLMPreprocessor.collate_fn`` uses for
    the collated token sequence (see ``scripts/gate_teacher_forced.py``'s
    printed batch keys: ``seqs``, ``loss_masks``, ``discrete_audio_indices``,
    ``discrete_audio_feats``, ``discrete_audio_lengths``). ``vocab_intervals``
    is not itself part of the collated batch dict (it lives on the
    ``SpeechLMJobTemplate``/model instead), so it must be passed separately.

    See the module docstring: on a raw (pre-forward) real ``collate_fn``
    batch, this raises ``ValueError`` - use
    ``batch["discrete_audio_indices"][:, 1]`` instead in that case.
    """
    return compute_audio_offsets(batch["seqs"], vocab_intervals)
