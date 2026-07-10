"""Utterance-boundary window selection over a session's turn timeline.

Pure module: no I/O, no torch.  Windows are cut at utterance boundaries,
CoVoMix-style (Fisher segmentation, arXiv:2404.06690 Algorithm 1, inherited
by CoVoMix2): a time instant ``t`` is an *eligible boundary* iff every merged
turn on every channel ends at least ``boundary_guard`` before ``t`` or starts
at least ``boundary_guard`` after it.  With ``boundary_guard = 0`` this
reduces to "no turn strictly contains ``t``" - turns touching ``t`` exactly
at their start or end are allowed, so zero-gap speaker exchanges are valid
cut points.  Activity is defined over merged turn spans (supersets of their
supervisions), so a cut can never split a merged turn through one of its
internal sub-``merge_gap`` gaps.

The session boundaries ``0.0`` and ``duration`` are always eligible: no
speech can cross a file edge (supervisions are clamped to the recording).
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Sequence

from .sssd import Recording, Turn

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
    dropped_span_sec: float = 0.0  # spans with no eligible boundary in reach
    dropped_tail_sec: float = 0.0
    dropped_empty_windows: int = 0
    # All-channel gap (next turn start - previous turn end) at each chosen
    # interior cut point; 0.0 for a zero-gap speaker exchange.
    cut_gaps: list[float] = field(default_factory=list)

    def merge(self, other: "WindowingStats") -> None:
        self.n_windows += other.n_windows
        self.dropped_span_sec += other.dropped_span_sec
        self.dropped_tail_sec += other.dropped_tail_sec
        self.dropped_empty_windows += other.dropped_empty_windows
        self.cut_gaps.extend(other.cut_gaps)


def blocked_intervals(
    turns: Sequence[Turn], boundary_guard: float
) -> list[tuple[float, float]]:
    """OPEN intervals where no boundary may fall: (start - g, end + g) per turn.

    Intervals are merged only where their interiors overlap; intervals that
    merely touch (``a.end + g == b.start - g``) stay separate so the touching
    instant remains eligible (that is exactly the zero-gap exchange case when
    ``g == 0``).
    """
    if boundary_guard < 0:
        raise ValueError(f"boundary_guard must be >= 0, got {boundary_guard}")
    spans = sorted((t.start - boundary_guard, t.end + boundary_guard) for t in turns)
    merged: list[tuple[float, float]] = []
    for a, b in spans:
        if merged and a < merged[-1][1]:  # strict interior overlap of open intervals
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def is_eligible_boundary(blocked: Sequence[tuple[float, float]], t: float) -> bool:
    """True iff ``t`` falls inside no blocked open interval."""
    return not any(a < t < b for a, b in blocked)


def first_eligible_boundary(blocked: Sequence[tuple[float, float]], t: float) -> float:
    """Smallest eligible instant >= ``t`` (blocked intervals sorted by start)."""
    for a, b in blocked:
        if a >= t:
            break
        if t < b:
            t = b
    return t


def select_window_spans(
    blocked: Sequence[tuple[float, float]],
    duration: float,
    *,
    window_min: float,
    window_max: float,
    tail_min: float,
    rng: random.Random,
) -> tuple[list[tuple[float, float]], WindowingStats]:
    """Greedy left-to-right span selection against the blocked-interval list.

    Each window draws a target duration uniform in [window_min, window_max]
    and extends to the FIRST eligible boundary at/after the target (CoVoMix
    Algorithm 1).  If that boundary lies beyond ``window_max`` the span up to
    it is dropped whole rather than emitted oversize.  The final remainder
    (<= window_max by construction) is emitted iff >= tail_min; this tail is
    the only window allowed below window_min.  Both edges of every emitted
    window are eligible boundaries (0 and ``duration`` count as eligible).
    """
    if not (0 < window_min <= window_max):
        raise ValueError(
            f"need 0 < window_min <= window_max, got {window_min}, {window_max}"
        )
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
        cut = min(first_eligible_boundary(blocked, cur + target), duration)
        if cut - cur <= window_max + _EPS:
            spans.append((cur, cut))
            stats.n_windows += 1
        else:
            stats.dropped_span_sec += cut - cur
        cur = cut
    return spans, stats


def _gap_at(turns: Sequence[Turn], t: float) -> float | None:
    """All-channel gap around an eligible boundary: next turn start minus
    previous turn end (0.0 for a zero-gap exchange); None at session edges."""
    before = [x.end for x in turns if x.end <= t + _EPS]
    after = [x.start for x in turns if x.start >= t - _EPS]
    if not before or not after:
        return None
    return max(0.0, min(after) - max(before))


def build_windows(
    session_id: str,
    rec: Recording,
    turns: Sequence[Turn],
    *,
    window_min: float,
    window_max: float,
    boundary_guard: float,
    tail_min: float,
    rng: random.Random,
) -> tuple[list[WindowRecord], WindowingStats]:
    """Window one session: boundary selection, turn assignment, empty-window drop."""
    blocked = blocked_intervals(turns, boundary_guard)
    spans, stats = select_window_spans(
        blocked,
        rec.duration,
        window_min=window_min,
        window_max=window_max,
        tail_min=tail_min,
        rng=rng,
    )
    records: list[WindowRecord] = []
    edges: set[float] = set()
    for t0, t1 in spans:
        # Boundaries never fall strictly inside a turn, so any turn
        # overlapping the span is fully contained in it (a turn touching t1
        # at its start belongs to the next window, touching t0 at its end to
        # the previous one).
        inside = tuple(t for t in turns if t.start >= t0 - _EPS and t.end <= t1 + _EPS)
        if not inside:
            stats.n_windows -= 1
            stats.dropped_empty_windows += 1
            continue
        edges.update(edge for edge in (t0, t1) if _EPS < edge < rec.duration - _EPS)
        records.append(
            WindowRecord(
                window_id=f"{session_id}_w{len(records):05d}",
                session_id=session_id,
                audio_relpath=rec.audio_relpath,
                num_channels=rec.num_channels,
                sample_rate=rec.sample_rate,
                t0=round(t0, 6),
                t1=round(t1, 6),
                # Turn times are rounded exactly like t0/t1: cuts land exactly
                # on turn endpoints, whose float accumulation noise (e.g.
                # end=912.6400000000001 vs cut rounded to 912.64) would
                # otherwise leave a stored turn nominally outside its own
                # window.  This also matches what to_json serializes.
                turns=tuple(
                    dataclasses.replace(t, start=round(t.start, 6), end=round(t.end, 6))
                    for t in inside
                ),
            )
        )
    for edge in sorted(edges):
        gap = _gap_at(turns, edge)
        if gap is not None:
            stats.cut_gaps.append(gap)
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
