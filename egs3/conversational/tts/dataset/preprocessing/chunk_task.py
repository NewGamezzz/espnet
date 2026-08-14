"""Special-token conditioning chunk-task plan: dataclasses and JSON codec.

Pure module: no I/O, no torch.  A ``ChunkTaskPlan`` records what an online
window sampler decided to do for special-token conditioning training - all
spans are session-absolute seconds (same coordinate space as ``Turn.start``/
``Turn.end`` and ``WindowRecord.t0``/``t1``), never window-relative.  This
module must never import ``windows.py``: ``windows.py`` imports FROM here
(``WindowRecord`` carries an optional ``ChunkTaskPlan``), and a reverse import
would create a cycle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .sssd import Turn

if TYPE_CHECKING:
    # Only for the ``record`` annotation below; never imported at runtime -
    # windows.py imports FROM chunk_task.py, so a real import here would
    # cycle. draw_chunk_task only ever touches record.t0/record.t1, so any
    # duck-typed object with those two attributes works.
    from .windows import WindowRecord

_KINDS = frozenset({"full", "prompt_only"})


@dataclass(frozen=True)
class ChunkTaskParams:
    """Sampling knobs for the special-token chunk task (values TBD by the
    training design; defaults here are placeholders, not ratified live
    settings)."""

    chunk_task_prob: float = 0.0
    prompt_only_prob: float = 0.2
    chunk_window_min: float = 15.0
    chunk_window_max: float = 35.0
    prev_slice_min: float = 2.0
    prev_slice_max: float = 10.0
    prompt_slice_min: float = 3.0
    prompt_slice_max: float = 8.0
    prompt_speech_floor: float = 3.0


@dataclass(frozen=True)
class ChunkTaskPlan:
    """One sampled chunk-task decision for a window.

    ``kind`` selects between a "full" plan (a <prev_chunk> slice precedes the
    window) and a "prompt_only" plan (no previous-chunk conditioning).
    ``prev_span`` is session-absolute and present iff ``kind == "full"``.
    ``prompt_spans`` has exactly one span per ORIGINAL channel (i.e. every
    speaker gets a voice reference - see the "prompt must cover all speakers"
    rule), all of equal length so the prompt batches as a fixed-size tensor.
    """

    kind: str  # "full" | "prompt_only"
    prev_span: tuple[float, float] | None  # session-absolute; None iff prompt_only
    prompt_spans: tuple[
        tuple[float, float], ...
    ]  # one per ORIGINAL channel, equal lengths

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {self.kind!r}")
        has_prev = self.prev_span is not None
        if has_prev != (self.kind == "full"):
            raise ValueError(
                "prev_span is None iff kind == 'prompt_only': got "
                f"kind={self.kind!r}, prev_span={self.prev_span!r}"
            )
        if has_prev and not (self.prev_span[0] < self.prev_span[1]):
            raise ValueError(f"prev_span must have start < end, got {self.prev_span!r}")
        if not self.prompt_spans:
            raise ValueError("prompt_spans must be non-empty")
        lengths = set()
        for start, end in self.prompt_spans:
            if not (start < end):
                raise ValueError(
                    f"prompt span must have start < end, got {(start, end)!r}"
                )
            lengths.add(round(end - start, 6))
        if len(lengths) != 1:
            raise ValueError(
                f"all prompt_spans must have equal (rounded) length, "
                f"got lengths {lengths}"
            )


def plan_to_json(p: ChunkTaskPlan) -> dict:
    return {
        "kind": p.kind,
        "prev_span": list(p.prev_span) if p.prev_span is not None else None,
        "prompt_spans": [list(span) for span in p.prompt_spans],
    }


def plan_from_json(d: dict) -> ChunkTaskPlan:
    prev_span = d["prev_span"]
    return ChunkTaskPlan(
        kind=d["kind"],
        prev_span=tuple(prev_span) if prev_span is not None else None,
        prompt_spans=tuple(tuple(span) for span in d["prompt_spans"]),
    )


def assembled_duration(record: "WindowRecord") -> float:
    """Total seconds ``record_costs`` should price ``record`` at.

    Duck-typed: uses only ``record.t0``/``record.t1``/``record.chunk_task``, so
    any object shaped like a ``WindowRecord`` works (see ``draw_chunk_task``'s
    docstring for the same convention). For a record with no chunk-task plan
    (the overwhelming majority - see ``chunk_task_prob``) this is plain
    ``t1 - t0``. For a chunk-task record, training assembles the window
    together with its prompt span (one representative span from
    ``prompt_spans`` - all spans share the same length, see
    ``ChunkTaskPlan.__post_init__``) and, for a "full" plan, its previous-chunk
    slice, so the packer must price the ASSEMBLED length, not just the window.
    """
    window_len = record.t1 - record.t0
    chunk_task = record.chunk_task
    if chunk_task is None:
        return window_len
    prompt_start, prompt_end = chunk_task.prompt_spans[0]
    lp = prompt_end - prompt_start
    if chunk_task.prev_span is not None:
        prev_start, prev_end = chunk_task.prev_span
        prev_len = prev_end - prev_start
    else:
        prev_len = 0.0
    return lp + prev_len + window_len


def _outside(turn: Turn, span: tuple[float, float]) -> bool:
    """True iff ``turn`` does not overlap ``span`` at all; touching exactly at
    an edge (``turn.end == span[0]`` or ``turn.start == span[1]``) counts as
    outside, matching the half-open convention used for eligible boundaries
    elsewhere in this recipe (see ``windows.blocked_intervals``)."""
    return turn.end <= span[0] or turn.start >= span[1]


def _min_extent(
    channel_turns: Sequence[Turn], anchor_start: float, floor: float
) -> float | None:
    """Minimal extent from ``anchor_start`` accumulating >= ``floor`` seconds
    of this channel's speech (sum of clipped turn overlaps).  ``channel_turns``
    is every turn of the anchor's channel, start-sorted; silence between turns
    (or another channel's speech) lengthens the extent without contributing to
    the floor.  None if the channel's turns from ``anchor_start`` onward never
    reach the floor (the channel runs out of speech first)."""
    speech = 0.0
    for turn in channel_turns:
        if turn.start < anchor_start:
            continue  # already behind the anchor; only forward turns count
        duration = turn.end - turn.start
        if speech + duration >= floor:
            # Floor is reached partway through (or exactly at the start of)
            # this turn: clip to the exact offset, not the whole turn.
            return (turn.start - anchor_start) + (floor - speech)
        speech += duration
    return None


def _max_extent(
    anchor_start: float,
    forbidden: tuple[float, float],
    exclusion_spans: Sequence[tuple[float, float]],
    session_duration: float,
    prompt_slice_max: float,
) -> float:
    """Maximal extent from ``anchor_start``: capped at ``prompt_slice_max``
    and at the nearest boundary the extent must not cross - the forbidden
    region ``F`` (if the anchor precedes it) or the session end (if it
    follows), and any exclusion span starting after the anchor."""
    if anchor_start < forbidden[0]:
        boundary = forbidden[0] - anchor_start
    else:
        boundary = session_duration - anchor_start
    for span_start, _span_end in exclusion_spans:
        if span_start > anchor_start:
            boundary = min(boundary, span_start - anchor_start)
    return min(prompt_slice_max, boundary)


def draw_chunk_task(
    record: "WindowRecord",
    session_turns: Sequence[Turn],
    session_duration: float,
    exclusion_spans: Sequence[tuple[float, float]],
    num_channels: int,
    rng: random.Random,
    params: ChunkTaskParams,
) -> tuple[ChunkTaskPlan | None, bool]:
    """Draw one seeded chunk-task plan for ``record``'s window (spec sections
    3-4).  ``record`` is duck-typed: only ``record.t0``/``record.t1`` are used,
    so callers may pass a ``WindowRecord`` or anything shaped like one.
    ``session_turns`` is the WHOLE session's turns (not just the window's),
    since prompt/prev material is drawn from anywhere in the session outside
    the window and outside ``exclusion_spans``.

    ``rng`` is consumed in a fixed order given the same inputs (draws are data-
    dependent only through the ``prompt_only`` branch: the prev-slice length is
    drawn iff step 1 didn't already pick prompt-only) - the ``prompt_only``
    draw, then the prev-slice length if attempted, then always the
    prompt-slice length, then one ``rng.sample`` shuffle per channel in
    channel order - so two calls with equivalently-seeded ``random.Random``
    instances and identical arguments always agree.

    Returns a ``(plan, degraded)`` tuple. ``plan`` is None when the window
    cannot support a task under these params (e.g. a channel has no eligible
    prompt anchor, or the per-channel floor and headroom requirements
    conflict); the caller counts this and falls back to an ordinary infill
    window. ``degraded`` is True iff step 1's coin picked "full" but step 2's
    H-clamp (insufficient session prefix / heavy exclusion before the window)
    forced the plan down to "prompt_only"; it is always False when the coin
    itself picked "prompt_only", and it stays meaningful even when ``plan`` is
    later None (a degrade can still be followed by a step-4/5 failure).
    """
    t0, t1 = record.t0, record.t1

    # Step 1: prompt-only coin flip.
    prompt_only = rng.random() < params.prompt_only_prob

    # Step 2: previous-chunk slice, only attempted when not already
    # prompt-only. Degrading here (short session prefix / heavy exclusion)
    # still discards prev_span, but the rng draw already happened - that is
    # intentional, see the docstring's fixed rng-order guarantee.
    degraded = False
    prev_start: float | None = None
    if not prompt_only:
        length = rng.uniform(params.prev_slice_min, params.prev_slice_max)
        prev_start = max(0.0, t0 - length)
        # Clamp up by the end of any exclusion span overlapping
        # (prev_start, t0); iterate to a fixed point since clamping past one
        # span's end can land inside another span that starts later.
        moved = True
        while moved:
            moved = False
            for es, ee in exclusion_spans:
                if es < t0 and ee > prev_start:
                    prev_start = max(prev_start, ee)
                    moved = True
        if t0 - prev_start < params.prev_slice_min:
            prompt_only = True
            prev_start = None
            degraded = True

    # Step 3: prompt-slice length draw (always happens) and the forbidden
    # region prompt anchors must avoid.
    lp_draw = rng.uniform(params.prompt_slice_min, params.prompt_slice_max)
    f_start = prev_start if prev_start is not None else t0
    forbidden = (f_start, t1)

    # Step 4: per-channel anchor search.
    anchors: list[float] = []
    m_values: list[float] = []
    a_values: list[float] = []
    for c in range(num_channels):
        channel_turns = sorted(
            (turn for turn in session_turns if turn.channel == c), key=lambda t: t.start
        )
        candidates = [
            turn
            for turn in channel_turns
            if _outside(turn, forbidden)
            and all(_outside(turn, es) for es in exclusion_spans)
        ]
        shuffled = rng.sample(candidates, len(candidates))
        chosen = None
        for anchor in shuffled:
            m_c = _min_extent(channel_turns, anchor.start, params.prompt_speech_floor)
            a_c = _max_extent(
                anchor.start,
                forbidden,
                exclusion_spans,
                session_duration,
                params.prompt_slice_max,
            )
            if m_c is not None and m_c <= a_c:
                chosen = (anchor.start, m_c, a_c)
                break
        if chosen is None:
            return None, degraded
        anchor_start, m_c, a_c = chosen
        anchors.append(anchor_start)
        m_values.append(m_c)
        a_values.append(a_c)

    # Step 5: pick one length shared by every channel's prompt span.
    min_a = min(a_values)
    max_m = max(m_values)
    lp = min(lp_draw, min_a)
    lp = max(lp, max_m)
    if lp > min_a:
        # The tightest channel's floor requirement exceeds another channel's
        # headroom: no shared length satisfies every channel at once.
        return None, degraded
    # Round both the shared length and each anchor onto the same 1e-6 grid
    # before adding them: anchors come from session_turns, which - unlike
    # WindowRecord.turns - is not guaranteed rounded (see windows.py's
    # to_json comment on turn-time float noise). Without this, two channels'
    # (anchor + lp) sums can round to lengths 1e-6 apart when lp sits near a
    # rounding tie, which ChunkTaskPlan.__post_init__ rejects as unequal.
    lp = round(lp, 6)
    prompt_spans = tuple(
        (start, round(start + lp, 6))
        for start in (round(anchor, 6) for anchor in anchors)
    )

    # Step 6: assemble the plan.
    kind = "prompt_only" if prompt_only else "full"
    prev_span = (round(prev_start, 6), t0) if prev_start is not None else None
    return (
        ChunkTaskPlan(kind=kind, prev_span=prev_span, prompt_spans=prompt_spans),
        degraded,
    )
