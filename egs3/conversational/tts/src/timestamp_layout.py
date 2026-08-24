"""Turn timelines for ``text_format: timestamps`` inference.

CoVoMix2 turns carry ORDINAL start/end (see ``external_testset``), so the
chunked path synthesizes a timeline (``synthesize_layout``): turns are laid
back-to-back on the frame grid with a fixed silence gap between consecutive
turns and never overlap, each turn as long as the duration policy's
per-turn estimate.  Frames are the unit of truth here - seconds are derived
as ``frame / fps`` so ``turn_frame_spans``' ``round()`` recovers the exact
frame - and consecutive chunk spans tile the timeline with no seam
arithmetic.  The SSSD ``generate`` path has real timestamps and only needs
the prompt blocks and the window turns placed on one sequence timeline
(``prompt_window_layout``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessing.text import FRAMES_PER_SECOND


def _as_turn(turn, start: float, end: float) -> Turn:
    if isinstance(turn, Turn):
        return dataclasses.replace(turn, start=start, end=end)
    return Turn(turn.channel, turn.speaker, turn.text, start, end)


@dataclass(frozen=True)
class TimestampLayout:
    turns: list[Turn]
    turn_frames: list[int]
    gap_frames: int
    fps: float

    def _start_frame(self, i: int) -> int:
        return sum(self.turn_frames[:i]) + i * self.gap_frames

    def chunk_span(self, a: int, b: int) -> tuple[int, int]:
        start = self._start_frame(a)
        end = self._start_frame(b - 1) + self.turn_frames[b - 1] + self.gap_frames
        return start, end - start


def synthesize_layout(
    turns: Sequence, turn_secs: Sequence[float], *, gap_sec: float,
    fps: float = FRAMES_PER_SECOND,
) -> TimestampLayout:
    if gap_sec < 0:
        raise ValueError(f"gap_sec must be >= 0, got {gap_sec}")
    if len(turns) != len(turn_secs):
        raise ValueError(f"{len(turns)} turns but {len(turn_secs)} durations")
    gap_frames = int(round(gap_sec * fps))
    placed, frames, cursor = [], [], 0
    for turn, sec in zip(turns, turn_secs):
        f = int(round(sec * fps))
        if f < 1 + len(turn.text):
            raise ValueError(
                f"turn block does not fit: needs {1 + len(turn.text)} frames, "
                f"estimate gives {f} ({turn.text!r})"
            )
        placed.append(_as_turn(turn, cursor / fps, (cursor + f) / fps))
        frames.append(f)
        cursor += f + gap_frames
    return TimestampLayout(turns=placed, turn_frames=frames, gap_frames=gap_frames, fps=fps)


def prompt_window_layout(
    prompt_turns: Sequence, prompt_block_samples: Sequence[int],
    window_turns: Sequence, window_t0: float, *, fs: int,
) -> list[Turn]:
    out, offset = [], 0
    for turn, samples in zip(prompt_turns, prompt_block_samples):
        out.append(_as_turn(turn, offset / fs, (offset + samples) / fs))
        offset += samples
    prompt_sec = offset / fs
    for turn in window_turns:
        start = prompt_sec + max(turn.start - window_t0, 0.0)
        out.append(_as_turn(turn, start, prompt_sec + (turn.end - window_t0)))
    return out
