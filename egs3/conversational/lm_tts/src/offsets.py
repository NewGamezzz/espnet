"""Per-row audio-offset extraction for a collated SpeechLM batch (Task 6).

Two functions, two different data sources, on purpose - see the Task 6
finding below for why one cannot simply delegate to the other.

``compute_audio_offsets_from_batch`` is what step-3 training actually hands
to the branch-exchange context's ``ctx.branches(align_offsets=...)`` (see
``src/branch_exchange/inject.py``): TAC must be exactly identity on every
pre-audio position, and ``ctx.branches`` needs, per row, the index where that
row's audio region starts - computed BEFORE the forward pass, since it has
to be ready to build the ``ctx.branches(...)`` context that wraps the model
call.

``compute_audio_offsets`` is the lower-level primitive: given a token
tensor, find the first stream-0 position whose value falls in a
``discrete_audio`` vocab interval. That only works when the tensor already
carries real per-stream codec ids in its audio region (hand-built tensors,
or a ``seqs`` snapshot captured AFTER the model's ``_embed`` step runs - see
below). It does NOT work on a fresh ``SpeechLMPreprocessor.collate_fn``
batch.

IMPORTANT (Task 6 finding, reproduced in
``tests/test_preprocessing_parity.py``'s ``TestRealPreprocessorParity``): on
BagPiper's real ``collate_fn`` output, ``compute_audio_offsets``'s stream-0
scan correctly finds NOTHING and raises ``ValueError`` on every real
training row. That is not a bug in the scan - it reflects how the pipeline
actually works. ``espnet2/speechlm/model/speechlm/lm/parallel.py``'s
``ParallelHFModel._embed`` encodes the raw waveform
(``discrete_audio_feats``) and writes the resulting codec ids into ``seqs``
IN PLACE, keyed by ``discrete_audio_indices``, only during the model forward
pass; before that, the audio region of a freshly collated ``seqs`` is the
``<|pad|>`` (id 0) placeholder, indistinguishable BY VALUE from ordinary
padding. There is no token-value signal pre-forward for a scan to find.

There IS a structural signal pre-forward, though: ``discrete_audio_indices``
(rows of ``(batch_idx, start, length)``), which the preprocessor computes
from role/modality/content token COUNTS (``speechlm_job.py``'s
``preprocessing``, step 3.4's ``accum_length``) - entirely independent of
the (not-yet-tokenized) audio content. That is the real pre-forward ground
truth, and it is what ``compute_audio_offsets_from_batch`` reads. This is
also why it takes only ``batch`` (matching the task's specified
single-argument signature) and never needs ``vocab_intervals``: it never
inspects token values at all.
"""

from __future__ import annotations

import torch


def compute_audio_offsets(seqs: torch.Tensor, vocab_intervals: dict) -> torch.LongTensor:
    """Per-row index of the FIRST position whose stream-0 token falls in any
    ``vocab_intervals["discrete_audio"]`` interval.

    See the module docstring: this only finds a real offset on a tensor
    whose audio region already carries real per-stream codec ids (hand-built
    tensors, or a post-``_embed`` snapshot) - NOT on a fresh
    ``collate_fn`` batch, where it correctly raises. Use
    ``compute_audio_offsets_from_batch`` for the pre-forward training case.

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


def compute_audio_offsets_from_batch(batch: dict) -> torch.LongTensor:
    """Per-row first audio-token position, read from a collated batch.

    This is the function step-3 training actually calls to build
    ``ctx.branches(align_offsets=...)``: unlike ``compute_audio_offsets``, it
    reads the preprocessor's own structural ``discrete_audio_indices``
    record (rows of ``(batch_idx, start, length)`` - see
    ``scripts/gate_teacher_forced.py``'s printed batch keys) instead of
    scanning token values, so it works on a real, pre-forward
    ``SpeechLMPreprocessor.collate_fn`` batch where the audio region is
    still the ``<|pad|>`` placeholder (see the module docstring's Task 6
    finding). If a row has more than one ``discrete_audio_indices`` entry,
    the offset is the MINIMUM (earliest) start recorded for that row.

    Args:
        batch: A ``collate_fn`` output dict. Must contain ``"seqs"`` (used
            only for its batch dimension) and ``"discrete_audio_indices"``.

    Returns:
        ``torch.LongTensor`` of shape ``(B,)``, one offset per row, on the
        same device as ``batch["discrete_audio_indices"]``.

    Raises:
        KeyError: ``batch`` is missing ``"seqs"`` or ``"discrete_audio_indices"``.
        ValueError: any row has no ``discrete_audio_indices`` entry (a
            training row must have audio).
    """
    seqs = batch["seqs"]
    dai = batch["discrete_audio_indices"]

    num_rows = seqs.shape[0]
    starts = torch.full((num_rows,), -1, dtype=torch.long, device=dai.device)
    for row in dai.tolist():
        bidx, start, _length = row
        if starts[bidx] == -1 or start < starts[bidx]:
            starts[bidx] = start

    missing = (starts == -1).nonzero(as_tuple=True)[0].tolist()
    if missing:
        raise ValueError(
            f"row(s) {missing} have no discrete_audio_indices entry; every "
            "training row must have audio"
        )
    return starts
