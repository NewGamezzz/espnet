"""Chunked ``infer`` stage for the external (CoVoMix2) test set.

Splits each dialogue's turn list at turn boundaries into chunks and runs one
ODE call per chunk: call k is conditioned on the FULL generated audio of
chunk k-1 (both channels) and the text of chunks k-1 and k, so every call
stays inside the short-generation regime where the model is grounded (the
full-run WER curve doubles per ~15 s of generated length).  Chunk k-1's
generated audio corresponds to known text BY CONSTRUCTION - chunks are whole
turns - so no aligner is ever needed.

Runs alongside ``src/external_inference.py`` exactly as that module runs
alongside the SSSD path: nothing here is imported by the other infer modes,
so ``generate_external`` and the SSSD modes stay bit-reproducible.

Chunk policy (config ``chunk:``, exactly one key):
* ``turns: N`` - every chunk is N consecutive turns (N=2 guarantees both
  speakers appear in every chunk given the test set's strict alternation).
* ``target_sec: S`` - pack turns until the chunk's PREDICTED duration would
  exceed S; a single turn longer than S still becomes its own oversized
  chunk (flagged in meta and counted in the log), never split inside.

Seams are hard concats: chunk k starts strictly after chunk k-1's audio, so
cross-seam overlap is structurally impossible - watch ``overlap_per_min``.

DETERMINISM CONTRACT - a documented departure from ``generate_external``:
sharding is BY DIALOGUE (a chunk chain cannot cross shards), and each shard
plans its round batches over its own dialogues, so outputs are a pure
function of the full config INCLUDING ``shard_count`` - rerunning any shard
reproduces it bit-for-bit, but changing ``shard_count`` redraws equally-valid
samples (exactly like changing the batching knobs does on the unchunked
path).  Fix ``shard_count`` in eval configs like any other sampling knob.
Round k reseeds with ``sampling.seed + k`` so consecutive chunks of the same
batch shape never reuse an identical noise draw.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.src.external_inference import (
    _load_prompt_wav,
    _probe_duration_sec,
    _prompt_blocks,
    _prompt_turns,
)
from egs3.conversational.tts.src.external_testset import (
    DEFAULT_DURATION_SCALE,
    ExternalRecord,
    assign_shard,
    duration_meta,
    load_covomix2_testset,
    plan_batches,
    select_records,
)
from egs3.conversational.tts.src.generation import (
    GenerationItem,
    build_preprocessor,
    generate_batch,
    load_model,
    load_vocoder,
    pad_branch_text,
    write_wav,
)

# Output-format helpers, reused verbatim so the infer paths can never drift
# apart in what they write (same import external_inference.py does).
from egs3.conversational.tts.src.inference import _reference_texts, _write_scp

logger = logging.getLogger(__name__)

MODE = "generate_external_chunked"


def estimate_turn_secs(
    record: ExternalRecord,
    prompt_seconds: Sequence[float],
    *,
    duration_scale: float,
    speed: float,
) -> list[float]:
    """Per-turn predicted seconds under the F5 per-speaker rate rule.

    The same rule as :func:`external_testset.estimate_duration_sec`, split
    per turn instead of summed: each turn is priced at its own speaker's
    prompt-measured seconds-per-character.  ``sum(estimate_turn_secs(...))``
    equals ``estimate_duration_sec(...)`` by construction, so chunking never
    changes the duration policy - only where the ODE calls are cut.
    """
    if len(prompt_seconds) != record.num_channels:
        raise ValueError(
            f"{record.dialogue_id}: got {len(prompt_seconds)} prompt durations "
            f"for {record.num_channels} channels"
        )
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    rates = []
    for ch, (prompt_sec, prompt) in enumerate(zip(prompt_seconds, record.prompts)):
        prompt_chars = len(prompt.text.encode("utf-8"))
        if prompt_sec <= 0 or prompt_chars <= 0:
            raise ValueError(
                f"{record.dialogue_id}: channel {ch} prompt is degenerate "
                f"({prompt_sec:.3f}s, {prompt_chars} chars)"
            )
        rates.append(prompt_sec / prompt_chars)
    return [
        len(turn.text.encode("utf-8"))
        * rates[turn.channel]
        * float(duration_scale)
        / float(speed)
        for turn in record.turns
    ]


def split_turns(
    turn_secs: Sequence[float],
    *,
    turns: int | None = None,
    target_sec: float | None = None,
) -> list[tuple[int, int]]:
    """Split turn indices into chunks; returns half-open ``(start, end)``
    ranges that partition ``range(len(turn_secs))`` in order.

    Exactly one policy must be given.  ``turns``: fixed count per chunk (the
    last chunk keeps the remainder).  ``target_sec``: greedy packing - a turn
    joins the current chunk unless that would push the chunk's predicted
    total past the target; a single turn longer than the target still gets
    its own chunk (the policy bounds growth, it never splits inside a turn).
    """
    if (turns is None) == (target_sec is None):
        raise ValueError("exactly one of `turns` / `target_sec` must be set")
    if not len(turn_secs):
        raise ValueError("no turns to split")
    if turns is not None:
        n = int(turns)
        if n < 1:
            raise ValueError(f"turns must be >= 1, got {turns}")
        return [(i, min(i + n, len(turn_secs))) for i in range(0, len(turn_secs), n)]
    target = float(target_sec)
    if target <= 0:
        raise ValueError(f"target_sec must be > 0, got {target_sec}")
    ranges: list[tuple[int, int]] = []
    start, acc = 0, 0.0
    for i, sec in enumerate(turn_secs):
        if i > start and acc + float(sec) > target:
            ranges.append((start, i))
            start, acc = i, 0.0
        acc += float(sec)
    ranges.append((start, len(turn_secs)))
    return ranges


def call_turns(
    record: ExternalRecord, ranges: Sequence[tuple[int, int]], k: int
) -> list[Turn]:
    """Turns conditioning ODE call ``k``.

    Call 0 is today's external layout: the per-channel prompt turns followed
    by chunk 0's turns.  Call k > 0 covers exactly the audio span of the
    call - chunk k-1 (the continuation prompt) then chunk k - which keeps
    the ``<turn>``/``<OTHER>`` budget consistent with the conditioning audio
    by construction.  Ranges are contiguous, so this is one slice.
    """
    if k == 0:
        a, b = ranges[0]
        return _prompt_turns(record) + list(record.turns[a:b])
    prev_start = ranges[k - 1][0]
    end = ranges[k][1]
    return list(record.turns[prev_start:end])


def run_chunked_inference(
    inference_config,
    *,
    training_config=None,
    model=None,
    vocoder=None,
) -> dict[str, Any]:
    """Execute the chunked external infer stage (implemented in a later task)."""
    raise NotImplementedError
