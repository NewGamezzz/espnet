"""Utterance-boundary window selection over a session's turn timeline.

Pure module: no I/O, no torch.  Windows are cut at utterance boundaries.
The *eligibility* rule follows CoVoMix's Fisher segmentation (arXiv:2404.06690,
inherited by CoVoMix2): a time instant ``t`` is an *eligible boundary* iff
every merged turn on every channel ends at least ``boundary_guard`` before
``t`` or starts at least ``boundary_guard`` after it.  With
``boundary_guard = 0`` this reduces to "no turn strictly contains ``t``" -
turns touching ``t`` exactly at their start or end are allowed, so zero-gap
speaker exchanges are valid cut points.  Activity is defined over merged turn
spans (supersets of their supervisions), so a cut can never split a merged
turn through one of its internal sub-``merge_gap`` gaps.

The *placement* search is ours (CoVoMix has no target duration at all): see
``select_window_spans`` for the hybrid closest-to-target rule with a restart
retry, which drops only genuinely oversized blocked spans, never audio
adjacent to them.

The session boundaries ``0.0`` and ``duration`` are always eligible: no
speech can cross a file edge (supervisions are clamped to the recording).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .chunk_task import ChunkTaskPlan, plan_from_json, plan_to_json
from .sssd import Recording, Turn, occupied_intervals

# Float-safety slack for containment checks on exact-boundary turns.
_EPS = 1e-9


def speaker_activity(
    turns: Sequence[Turn], num_channels: int
) -> tuple[int, tuple[float, ...], int]:
    """Per-window speaker-activity metadata for training-time filtering.

    Returns ``(num_active_speakers, channel_speech_sec, exchange_count)``:
    channels with at least one turn, per-channel speech seconds (sum of that
    channel's turn durations, rounded like other times), and the number of
    speaker alternations in the turns sorted by start time (0 for
    single-speaker windows).
    """
    speech = [0.0] * num_channels
    for t in turns:
        speech[t.channel] += t.end - t.start
    ordered = sorted(turns, key=lambda t: (t.start, t.channel))
    exchanges = sum(1 for a, b in zip(ordered, ordered[1:]) if a.channel != b.channel)
    num_active = sum(1 for s in speech if s > 0)
    return num_active, tuple(round(s, 6) for s in speech), exchanges


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
    # Special-token conditioning chunk-task plan; None for ordinary windows
    # (the overwhelming majority - see chunk_task_prob). Kept optional and
    # last-before-derived so unrelated call sites building WindowRecord
    # positionally/by keyword without this field are unaffected.
    chunk_task: "ChunkTaskPlan | None" = None
    # Derived from turns/num_channels, never passed in: always consistent
    # with the stored (rounded) turn times, including after from_json.
    num_active_speakers: int = field(init=False)
    channel_speech_sec: tuple[float, ...] = field(init=False)
    exchange_count: int = field(init=False)

    def __post_init__(self) -> None:
        active, speech_sec, exchanges = speaker_activity(self.turns, self.num_channels)
        object.__setattr__(self, "num_active_speakers", active)
        object.__setattr__(self, "channel_speech_sec", speech_sec)
        object.__setattr__(self, "exchange_count", exchanges)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class WindowingStats:
    n_windows: int = 0
    dropped_span_sec: float = 0.0  # oversized blocked spans (> window_max)
    dropped_tail_sec: float = 0.0
    # Prefixes shorter than tail_min discarded before a restart (see
    # select_window_spans step 2); kept separate from dropped_span_sec so the
    # summary distinguishes unbreakable audio from search slivers.
    dropped_sliver_sec: float = 0.0
    dropped_empty_windows: int = 0
    # Windows removed by the optional coverage guard (build_windows'
    # min_coverage / trim_to_turns knobs). Both stay 0 unless a knob is set,
    # so the default behavior - and these stats - are unchanged.
    dropped_low_coverage_windows: int = 0
    dropped_low_coverage_sec: float = 0.0
    dropped_trimmed_short_windows: int = 0
    dropped_trimmed_short_sec: float = 0.0
    # Seconds skipped by the optional snap_start_to_turn knob (silence/holes
    # between a cut and the next turn). Stays 0 unless the knob is set.
    snapped_gap_sec: float = 0.0
    # All-channel gap (next turn start - previous turn end) at each chosen
    # interior cut point; 0.0 for a zero-gap speaker exchange.
    cut_gaps: list[float] = field(default_factory=list)
    # Special-token conditioning chunk-task counters (planner.py). All stay 0
    # unless a recipe opts into chunk_params, so default behavior - and these
    # stats - are unchanged. n_chunk_full/n_chunk_prompt_only count attached
    # ChunkTaskPlan.kind values; n_chunk_degraded is the subset of
    # n_chunk_prompt_only where the prompt_only_prob coin picked "full" but
    # draw_chunk_task's H-clamp forced "prompt_only"; n_chunk_fallback_infill
    # counts windows where draw_chunk_task returned None and the window fell
    # back to an ordinary infill window.
    n_chunk_full: int = 0
    n_chunk_prompt_only: int = 0
    n_chunk_degraded: int = 0
    n_chunk_fallback_infill: int = 0

    def merge(self, other: "WindowingStats") -> None:
        self.n_windows += other.n_windows
        self.dropped_span_sec += other.dropped_span_sec
        self.dropped_tail_sec += other.dropped_tail_sec
        self.dropped_sliver_sec += other.dropped_sliver_sec
        self.dropped_empty_windows += other.dropped_empty_windows
        self.dropped_low_coverage_windows += other.dropped_low_coverage_windows
        self.dropped_low_coverage_sec += other.dropped_low_coverage_sec
        self.dropped_trimmed_short_windows += other.dropped_trimmed_short_windows
        self.dropped_trimmed_short_sec += other.dropped_trimmed_short_sec
        self.snapped_gap_sec += other.snapped_gap_sec
        self.cut_gaps.extend(other.cut_gaps)
        self.n_chunk_full += other.n_chunk_full
        self.n_chunk_prompt_only += other.n_chunk_prompt_only
        self.n_chunk_degraded += other.n_chunk_degraded
        self.n_chunk_fallback_infill += other.n_chunk_fallback_infill


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


def _containing_block(
    blocked: Sequence[tuple[float, float]], t: float
) -> tuple[float, float] | None:
    """The blocked open interval strictly containing ``t``, or None."""
    for a, b in blocked:
        if a >= t:
            break
        if t < b:
            return (a, b)
    return None


def _next_turn_start(turn_starts: Sequence[float], t: float) -> float | None:
    """Smallest turn start >= ``t`` (``turn_starts`` ascending), or None."""
    for s in turn_starts:
        if s >= t - _EPS:
            return s
    return None


def select_window_spans(
    blocked: Sequence[tuple[float, float]],
    duration: float,
    *,
    window_min: float,
    window_max: float,
    tail_min: float,
    rng: random.Random,
    turn_starts: Sequence[float] = (),
    snap_start_to_turn: bool = False,
) -> tuple[list[tuple[float, float]], WindowingStats]:
    """Greedy left-to-right span selection against the blocked-interval list.

    With ``snap_start_to_turn`` (off by default), each iteration first advances
    ``cur`` to the next turn start (``turn_starts`` ascending), skipping any
    silence or transcription hole after the previous cut so the length budget is
    spent on speech; the skipped seconds are recorded in ``snapped_gap_sec``.
    Starting a window exactly on a turn start never splits a turn, so it is valid
    even when ``boundary_guard`` would place that instant inside a blocked band.

    Each iteration draws a target duration uniform in [window_min, window_max]
    and applies a retry loop:

    1. Hybrid cut: cut at the eligible boundary in
       ``[cur + window_min, cur + window_max]`` closest to the target (ties
       toward the earlier one).
    2. Restart: if no such boundary exists, exactly one blocked interval
       ``(bs, be)`` covers the whole search range.  When ``bs`` lies ahead of
       ``cur``, emit the prefix ``[cur, bs]`` as a mini-window if it is at
       least ``tail_min`` (mid-session windows in [tail_min, window_min) are
       legal), otherwise count it as a dropped sliver; then restart from
       ``bs``, so a block no longer than ``window_max`` is captured inside an
       ordinary window.
    3. Oversized drop: if the covering block starts at ``cur`` itself, it is
       longer than ``window_max`` from here, so drop exactly ``[cur, be]`` -
       never the audio adjacent to it.

    The final remainder (<= window_max) is emitted iff >= tail_min.  Both
    edges of every emitted window are eligible boundaries (0 and ``duration``
    count as eligible).  Every branch strictly advances ``cur``, so the loop
    terminates.
    """
    if not (0 < window_min <= window_max):
        raise ValueError(
            f"need 0 < window_min <= window_max, got {window_min}, {window_max}"
        )
    stats = WindowingStats()
    spans: list[tuple[float, float]] = []
    cur = 0.0
    while cur < duration - _EPS:
        if snap_start_to_turn:
            # Begin the next window on the next turn, skipping the intervening
            # silence/hole; strictly non-decreasing, so the loop still advances.
            nxt = _next_turn_start(turn_starts, cur)
            if nxt is None or nxt >= duration - _EPS:
                break
            stats.snapped_gap_sec += nxt - cur
            cur = nxt
        remaining = duration - cur
        if remaining <= window_max + _EPS:
            if remaining >= tail_min:
                spans.append((cur, duration))
                stats.n_windows += 1
            else:
                stats.dropped_tail_sec += remaining
            break
        lo, hi = cur + window_min, cur + window_max
        target = cur + rng.uniform(window_min, window_max)
        block = _containing_block(blocked, target)
        if block is None:
            cut = target  # the target itself is eligible
        else:
            a, b = block
            # The nearest eligible instants to a blocked target are exactly
            # the containing block's edges; keep those inside [lo, hi].
            cands = [c for c in (a, b) if lo - _EPS <= c <= hi + _EPS]
            if cands:
                cut = min(cands, key=lambda c: (abs(c - target), c))
            elif a > cur + _EPS:
                # (a, b) covers [lo, hi]: retry from its start edge, emitting
                # the prefix as a mini-window when it is long enough.
                if a - cur >= tail_min:
                    spans.append((cur, a))
                    stats.n_windows += 1
                else:
                    stats.dropped_sliver_sec += a - cur
                cur = a
                continue
            else:
                # The block starts at cur and extends past cur + window_max:
                # genuinely oversized, drop it and nothing else.
                end = min(b, duration)
                stats.dropped_span_sec += end - cur
                cur = end
                continue
        spans.append((cur, cut))
        stats.n_windows += 1
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
    trim_to_turns: bool = False,
    min_coverage: float = 0.0,
    snap_start_to_turn: bool = False,
) -> tuple[list[WindowRecord], WindowingStats]:
    """Window one session: boundary selection, turn assignment, empty-window drop.

    Three optional guards strip transcription holes - stretches of real audible
    speech that carry no turn (SSSD's Parakeet pseudo-labels have such gaps).
    All default to exact no-ops, so the shared windowing stays byte-compatible
    with the bagpiper copy unless a recipe opts in.

    ``snap_start_to_turn`` begins each window on the next turn instead of the
    previous cut, skipping the silence/hole between them so the length budget
    covers speech (this brackets the *leading* edge during selection; the
    skipped seconds are counted in ``snapped_gap_sec``). ``trim_to_turns``
    shrinks each window to ``[min turn start, max turn end]`` (clamped inside the
    original span), losslessly removing lead-in/trail-out holes - it also
    brackets the *trailing* edge that snap-start leaves; a window trimmed below
    ``tail_min`` is dropped. ``min_coverage`` is a conservative backstop for
    *interior* holes that neither reaches: a window whose transcribed union
    (``occupied_intervals``, so simultaneous speech on two channels counts once,
    not twice) covers less than ``min_coverage`` of its duration is dropped.
    Note a turn-coverage floor cannot tell a transcription hole from a genuinely
    silent pause and drops both, which is why trimming is the primary fix and
    this is only a floor.
    """
    blocked = blocked_intervals(turns, boundary_guard)
    turn_starts = sorted(t.start for t in turns)
    spans, stats = select_window_spans(
        blocked,
        rec.duration,
        window_min=window_min,
        window_max=window_max,
        tail_min=tail_min,
        rng=rng,
        turn_starts=turn_starts,
        snap_start_to_turn=snap_start_to_turn,
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
        # Optional hole guards. w_t0/w_t1 are the emitted window bounds; with
        # both knobs off they equal (t0, t1), so behavior is unchanged.
        w_t0, w_t1 = t0, t1
        if trim_to_turns:
            # min/max, not inside[0]/inside[-1]: turns are start-sorted, so the
            # last-starting turn is not the last-ending one. max(t0, ...) guards
            # float noise so trimming only ever shrinks the span.
            w_t0 = max(t0, min(t.start for t in inside))
            w_t1 = min(t1, max(t.end for t in inside))
            if w_t1 - w_t0 < tail_min:
                stats.n_windows -= 1
                stats.dropped_trimmed_short_windows += 1
                stats.dropped_trimmed_short_sec += w_t1 - w_t0
                continue
        if min_coverage > 0.0:
            covered = sum(b - a for a, b in occupied_intervals(inside))
            if covered < min_coverage * (w_t1 - w_t0):
                stats.n_windows -= 1
                stats.dropped_low_coverage_windows += 1
                stats.dropped_low_coverage_sec += w_t1 - w_t0
                continue
        edges.update(edge for edge in (w_t0, w_t1) if _EPS < edge < rec.duration - _EPS)
        records.append(
            WindowRecord(
                window_id=f"{session_id}_w{len(records):05d}",
                session_id=session_id,
                audio_relpath=rec.audio_relpath,
                num_channels=rec.num_channels,
                sample_rate=rec.sample_rate,
                t0=round(w_t0, 6),
                t1=round(w_t1, 6),
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
    d = {
        "window_id": w.window_id,
        "session_id": w.session_id,
        "audio_relpath": w.audio_relpath,
        "num_channels": w.num_channels,
        "sample_rate": w.sample_rate,
        "t0": w.t0,
        "t1": w.t1,
        "duration": round(w.t1 - w.t0, 6),
        "num_active_speakers": w.num_active_speakers,
        "channel_speech_sec": list(w.channel_speech_sec),
        "exchange_count": w.exchange_count,
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
    # Key omitted (not emitted as null) when absent: the golden-parity tests
    # byte-compare manifests produced before this field existed, and every
    # ordinary window has chunk_task=None, so a null key here would diverge
    # every golden line.
    if w.chunk_task is not None:
        d["chunk_task"] = plan_to_json(w.chunk_task)
    return d


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
        chunk_task=plan_from_json(d["chunk_task"]) if "chunk_task" in d else None,
    )


def write_window_manifest(path, records) -> int:
    """Write records as one JSON object per line (the manifest format
    ``dataset.read_window_manifest`` reads back).  Shared by corpus builders
    so every corpus emits the identical schema; returns the record count.

    Writes to a sibling ``.tmp`` path and ``os.replace``s it onto ``path`` so
    a build killed mid-write (e.g. a login-node time limit) never leaves a
    truncated file at ``path`` itself - builders' ``is_built`` is
    existence-only and would otherwise treat that truncation as built.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_json(record)) + "\n")
            n += 1
    os.replace(tmp_path, path)
    return n
