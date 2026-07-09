"""Silence-aligned window selection over a session's turn timeline.

Pure module: no I/O, no torch.  Cut points are placed only where no turn is
active on any channel for at least ``silence_min`` seconds, so no utterance is
ever truncated.  Occupied intervals come from *merged turn spans* (supersets
of their supervisions), which also guarantees a cut never splits a merged turn
through one of its internal sub-``merge_gap`` gaps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .sssd import Recording, Turn, occupied_intervals

# Float-safety slack for containment checks on exact-boundary turns.
_EPS = 1e-9


@dataclass(frozen=True)
class WindowRecord:
    window_id: str
    session_id: str
    audio_relpath: str
    num_channels: int
    sample_rate: int  # source rate (48000), not the training rate
    t0: float
    t1: float
    turns: tuple[Turn, ...]

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class WindowingStats:
    n_windows: int = 0
    dropped_span_sec: float = 0.0  # unwindowable long-overlap spans
    dropped_tail_sec: float = 0.0
    dropped_empty_windows: int = 0

    def merge(self, other: "WindowingStats") -> None:
        self.n_windows += other.n_windows
        self.dropped_span_sec += other.dropped_span_sec
        self.dropped_tail_sec += other.dropped_tail_sec
        self.dropped_empty_windows += other.dropped_empty_windows


def candidate_cut_points(
    occupied: Sequence[tuple[float, float]], duration: float, silence_min: float
) -> list[float]:
    """Cut candidates: midpoints of all-channel silences >= silence_min, plus
    the session boundaries 0.0 and ``duration`` (nothing is active across
    them; callers clamp supervisions to the recording first)."""
    cuts = {0.0, duration}
    prev_end = 0.0
    for start, end in occupied:
        if start - prev_end >= silence_min:
            cuts.add((prev_end + start) / 2.0)
        prev_end = max(prev_end, end)
    if duration - prev_end >= silence_min:
        cuts.add((prev_end + duration) / 2.0)
    return sorted(cuts)


def select_window_spans(
    cuts: Sequence[float],
    duration: float,
    *,
    window_min: float,
    window_max: float,
    tail_min: float,
    rng: random.Random,
) -> tuple[list[tuple[float, float]], WindowingStats]:
    """Greedy left-to-right span selection over the sorted cut list.

    Each window draws a target duration uniform in [window_min, window_max]
    and ends at the cut closest to it (ties -> earlier).  A stretch with no
    qualifying cut (e.g. sustained overlap) is dropped whole rather than
    emitted oversize.  The final remainder (<= window_max by construction) is
    emitted iff >= tail_min; this tail is the only window allowed below
    window_min.
    """
    if not (0 < window_min <= window_max):
        raise ValueError(f"need 0 < window_min <= window_max, got {window_min}, {window_max}")
    stats = WindowingStats()
    spans: list[tuple[float, float]] = []
    cur = 0.0
    while cur < duration - _EPS:
        remaining = duration - cur
        if remaining <= window_max + _EPS:
            if remaining >= tail_min:
                spans.append((cur, duration))
                stats.n_windows += 1
            else:
                stats.dropped_tail_sec += remaining
            break
        target = rng.uniform(window_min, window_max)
        cand = [c for c in cuts if cur + window_min <= c <= cur + window_max]
        if cand:
            cut = min(cand, key=lambda c: (abs(c - (cur + target)), c))
            spans.append((cur, cut))
            stats.n_windows += 1
            cur = cut
        else:
            nxt = min(c for c in cuts if c > cur + window_max)
            stats.dropped_span_sec += nxt - cur
            cur = nxt
    return spans, stats


def build_windows(
    session_id: str,
    rec: Recording,
    turns: Sequence[Turn],
    *,
    window_min: float,
    window_max: float,
    silence_min: float,
    tail_min: float,
    rng: random.Random,
) -> tuple[list[WindowRecord], WindowingStats]:
    """Window one session: cut selection, turn assignment, empty-window drop."""
    occupied = occupied_intervals(turns)
    cuts = candidate_cut_points(occupied, rec.duration, silence_min)
    spans, stats = select_window_spans(
        cuts,
        rec.duration,
        window_min=window_min,
        window_max=window_max,
        tail_min=tail_min,
        rng=rng,
    )
    records: list[WindowRecord] = []
    for t0, t1 in spans:
        # Cuts never intersect occupied intervals, so any turn overlapping the
        # span is fully contained in it.
        inside = tuple(t for t in turns if t.start >= t0 - _EPS and t.end <= t1 + _EPS)
        if not inside:
            stats.n_windows -= 1
            stats.dropped_empty_windows += 1
            continue
        records.append(
            WindowRecord(
                window_id=f"{session_id}_w{len(records):05d}",
                session_id=session_id,
                audio_relpath=rec.audio_relpath,
                num_channels=rec.num_channels,
                sample_rate=rec.sample_rate,
                t0=round(t0, 6),
                t1=round(t1, 6),
                turns=inside,
            )
        )
    return records, stats


def to_json(w: WindowRecord) -> dict:
    return {
        "window_id": w.window_id,
        "session_id": w.session_id,
        "audio_relpath": w.audio_relpath,
        "num_channels": w.num_channels,
        "sample_rate": w.sample_rate,
        "t0": w.t0,
        "t1": w.t1,
        "duration": round(w.t1 - w.t0, 6),
        "turns": [
            {
                "channel": t.channel,
                "speaker": t.speaker,
                "text": t.text,
                "start": round(t.start, 6),
                "end": round(t.end, 6),
            }
            for t in w.turns
        ],
    }


def from_json(d: dict) -> WindowRecord:
    return WindowRecord(
        window_id=d["window_id"],
        session_id=d["session_id"],
        audio_relpath=d["audio_relpath"],
        num_channels=int(d["num_channels"]),
        sample_rate=int(d["sample_rate"]),
        t0=float(d["t0"]),
        t1=float(d["t1"]),
        turns=tuple(
            Turn(
                channel=int(t["channel"]),
                speaker=t["speaker"],
                text=t["text"],
                start=float(t["start"]),
                end=float(t["end"]),
            )
            for t in d["turns"]
        ),
    )
