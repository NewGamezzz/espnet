"""Per-epoch window planning over session records (the online chunker).

Pure module: wraps windows.build_windows without modifying it.  Frozen mode
(epoch=None) seeds random.Random(f"{seed}:window:{sid}") - the exact stream
the retired offline builders used - so it reproduces their window manifests
bit-for-bit; that mode serves valid/test splits, inference, and the golden
parity tests.  Epoch mode appends ":epoch{epoch}" for fresh training windows
each epoch that any rank or worker can re-derive independently.

Special-token conditioning chunk-task mode (chunk_task.py) hooks into epoch
mode only: passing a ``chunk_params`` whose ``chunk_task_prob > 0`` draws one
per-SESSION-per-epoch coin from the SAME rng that windows itself, and on a
"chunk" outcome plans that session's windows against ``chunk_params``'s
window range instead of ``params``'s, then calls ``draw_chunk_task`` per
surviving window (same rng, after exclusion-span filtering) to attach a
``ChunkTaskPlan``.  Frozen mode, ``chunk_params=None``, ``chunk_task_prob ==
0``, and ``session.atomic`` all take the exact pre-chunk-task code path with
identical RNG construction and zero extra draws - the bit-parity guarantee
the golden tests in test_parity.py depend on.

Timestamp-aligned text mode (Mode T, text.py) hooks in similarly but through
its OWN dedicated rng stream, never the shared window/chunk-task rng above:
``timestamp_align_prob == 0`` (the default) or ``epoch is None`` takes the
exact pre-Mode-T code path - atomic early return, non-chunking return, and
chunk-task tail alike - with zero extra draws on the window/chunk-task rng,
so it never perturbs bit-parity.  When live, every surviving record (in
record order, atomic sessions included) draws one coin from
``random.Random(f"{seed}:textmode:{session_id}:epoch{epoch}")``; a heads coin
flags ``timestamp_text=True`` iff ``timestamp_fits`` passes, else the record
stays Mode O and the degrade counter increments.

Per-channel mask regimes (design 2026-08-15) hook in last, through their OWN
dedicated rng stream ``random.Random(f"{seed}:maskmode:{sid}:epoch{e}")``:
both probs 0 (the default) or ``epoch is None`` takes the exact prior code
path with zero extra draws. When live, every surviving record draws a
layered coin, context first: a context hit on an N >= 2 window attaches a
uniform proper subset ``context_channels`` (k ~ U{1..N-1}); a context hit on
an N = 1 window degrades (counted) and falls through to the independent
coin, which flags ``independent_mask``.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Iterable

from .chunk_task import ChunkTaskParams, draw_chunk_task
from .sessions import SessionRecord
from .sssd import Recording
from .text import timestamp_fits
from .windows import WindowingStats, WindowRecord, build_windows


@dataclass(frozen=True)
class WindowParams:
    """Training-time windowing knobs (ratified 2026-08-06 live values)."""

    # Training window target duration is drawn uniformly from
    # [window_min, window_max].
    window_min: float = 10.0
    # 80 s exceeds the F5 pretraining clip regime (< 30 s Emilia clips); chosen
    # deliberately to capture longer interactions - revisit if fine-tuning
    # quality degrades on long windows. Ratified 2026-08-06 from the live
    # Delta training build (previously a local override of the old 60 s
    # default). NOTE: a single 80 s N=2 window sets a ~40 GB activation floor
    # on A100-40G regardless of batch_bins.
    window_max: float = 80.0
    # Utterance-boundary cuts; eligibility follows CoVoMix's Fisher
    # segmentation (arXiv:2404.06690), the placement search is ours (hybrid
    # closest-to-target with restart retry, see windows.py). A boundary at t
    # is eligible iff every turn ends >= boundary_guard before t or starts >=
    # boundary_guard after it. 0.0 is faithful to CoVoMix (zero margin, human
    # LDC timestamps); SSSD timestamps are Parakeet pseudo-labels, so this
    # knob exists to reject boundaries where the neighboring utterance's
    # alignment jitter could leak un-covered speech into the window. May be
    # raised after inspecting debug dumps.
    boundary_guard: float = 0.0
    # Shortest emitted window below window_min: bounds both the session tail
    # and mid-session mini-windows emitted before an oversized blocked span.
    tail_min: float = 5.0
    # Transcription-hole guards (SSSD's Parakeet pseudo-labels leave
    # stretches of real speech unlabeled; a window that sweeps one in pairs
    # audio with a transcript that does not cover it - see windows.py).
    # trim_to_turns shrinks each window to [first turn start, last turn end],
    # losslessly removing lead-in/trail-out holes; a window trimmed below
    # tail_min is dropped. min_coverage is a conservative backstop for
    # interior holes: drop a window whose transcribed union covers less than
    # this fraction of its duration (0.0 disables). A turn-coverage floor
    # cannot distinguish a hole from a genuine silent pause, so keep it low.
    # Both guards ON, ratified 2026-08-06 from the live Delta training build
    # (previously local overrides of the off-by-default repo values).
    trim_to_turns: bool = True
    min_coverage: float = 0.0
    # snap_start_to_turn begins each window on the next turn instead of the
    # previous cut, skipping the silence/hole between them so the length
    # budget covers speech (raises coverage; skipped seconds are reported as
    # snapped-gaps). It brackets the leading edge during selection;
    # trim_to_turns brackets the trailing edge it leaves.
    snap_start_to_turn: bool = True
    # How exclusion spans (Fisher's unintelligible utterances, Chorus's
    # <UNKNOWN/> utterances) interact with windowing (design 2026-08-28):
    #   "drop" (default, bit-parity): plan as if there were no spans, then
    #         discard every window overlapping one - a 0.8 s span kills a
    #         whole 10-80 s window (measured: 21% of Fisher, 37% of Chorus).
    #   "cut": the spans become hard window boundaries - the session is
    #         planned segment by segment between them, so only the span
    #         itself (plus any turn that overlapped it, whose text would not
    #         cover that audio, and slivers too short to window) is lost.
    #         No untranscribed words ever enter a window.
    exclusion_mode: str = "drop"

    def __post_init__(self) -> None:
        if self.exclusion_mode not in ("drop", "cut"):
            raise ValueError(
                f"exclusion_mode must be 'drop' or 'cut', got {self.exclusion_mode!r}"
            )


def _window_rng(seed: int, session_id: str, epoch: int | None) -> random.Random:
    if epoch is None:
        return random.Random(f"{seed}:window:{session_id}")
    return random.Random(f"{seed}:window:{session_id}:epoch{epoch}")


def _textmode_rng(seed: int, session_id: str, epoch: int) -> random.Random:
    # Dedicated stream: Mode T coins must never perturb the window/chunk rng
    # (bit-parity), so they get their own Random keyed like _window_rng.
    return random.Random(f"{seed}:textmode:{session_id}:epoch{epoch}")


def _apply_timestamp_coin(
    records: list[WindowRecord],
    stats: WindowingStats,
    session: SessionRecord,
    seed: int,
    epoch: int | None,
    timestamp_align_prob: float,
) -> list[WindowRecord]:
    if timestamp_align_prob <= 0 or epoch is None or not records:
        return records
    rng = _textmode_rng(seed, session.session_id, epoch)
    out = []
    for record in records:
        if rng.random() < timestamp_align_prob:
            if timestamp_fits(record.turns, record.t0, record.t1):
                record = dataclasses.replace(record, timestamp_text=True)
                stats.n_timestamp_windows += 1
            else:
                stats.n_timestamp_degraded += 1
        out.append(record)
    return out


def _maskmode_rng(seed: int, session_id: str, epoch: int) -> random.Random:
    # Dedicated stream like _textmode_rng: mask coins must never perturb the
    # window/chunk-task or text-mode rngs (bit-parity).
    return random.Random(f"{seed}:maskmode:{session_id}:epoch{epoch}")


def _apply_mask_coin(
    records: list[WindowRecord],
    stats: WindowingStats,
    session: SessionRecord,
    seed: int,
    epoch: int | None,
    context_channel_prob: float,
    independent_mask_prob: float,
) -> list[WindowRecord]:
    """Per-window layered mask coin (design 2026-08-15), context first.

    Draw order per record is deterministic given the record sequence: one
    context coin (only if its prob > 0), then on a context hit with N >= 2
    one randint + one sample for the subset, else one independent coin (only
    if its prob > 0). Zero draws when both probs are 0 or epoch is None, so
    frozen mode and default configs never construct the rng at all.
    """
    if epoch is None or not records:
        return records
    if context_channel_prob <= 0 and independent_mask_prob <= 0:
        return records
    rng = _maskmode_rng(seed, session.session_id, epoch)
    out = []
    for record in records:
        ctx_hit = context_channel_prob > 0 and rng.random() < context_channel_prob
        if ctx_hit and record.num_channels >= 2:
            k = rng.randint(1, record.num_channels - 1)
            chans = tuple(sorted(rng.sample(range(record.num_channels), k)))
            record = dataclasses.replace(record, context_channels=chans)
            stats.n_context_windows += 1
        else:
            if ctx_hit:
                # A 1-channel window cannot spare a fully observed row.
                stats.n_context_degraded += 1
            if independent_mask_prob > 0 and rng.random() < independent_mask_prob:
                record = dataclasses.replace(record, independent_mask=True)
                stats.n_independent_windows += 1
        out.append(record)
    return out


def cut_segments(
    session: SessionRecord,
) -> tuple[list[tuple[float, float, tuple]], tuple[tuple[float, float], ...], int]:
    """Split a session at its exclusion spans (``exclusion_mode == "cut"``).

    Spans are merged, then extended over any turn that overlaps one (a turn
    merged across a dropped utterance, or one containing an inline
    unintelligible word): that turn's text does not cover that audio, so it
    is removed and its seconds join the excluded region.  Extension repeats
    until stable, so an excluded turn that overlaps another span merges too.

    Args:
        session: The session record (turns + exclusion spans).

    Returns:
        ``(segments, extended_spans, n_excluded_turns)`` where each segment
        is ``(start, end, turns_fully_inside)`` covering the gaps between the
        extended spans within ``[0, session.duration]``.
    """
    spans = sorted(
        (max(0.0, a), min(session.duration, b)) for a, b in session.exclusion_spans
    )
    merged: list[list[float]] = []
    for a, b in spans:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    turns = list(session.turns)
    excluded_turns: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i, t in enumerate(turns):
            if i in excluded_turns:
                continue
            for span in merged:
                if t.start < span[1] and span[0] < t.end:
                    excluded_turns.add(i)
                    span[0] = min(span[0], t.start)
                    span[1] = max(span[1], t.end)
                    changed = True
                    break
        if changed:
            merged.sort()
            out: list[list[float]] = []
            for a, b in merged:
                if out and a <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], b)
                else:
                    out.append([a, b])
            merged = out
    kept = [t for i, t in enumerate(turns) if i not in excluded_turns]
    gaps: list[tuple[float, float]] = []
    cur = 0.0
    for a, b in merged:
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if session.duration > cur:
        gaps.append((cur, session.duration))
    segments = []
    for a, b in gaps:
        inside = tuple(t for t in kept if t.start >= a - 1e-9 and t.end <= b + 1e-9)
        segments.append((a, b, inside))
    return segments, tuple((a, b) for a, b in merged), len(excluded_turns)


def _build_session_windows(
    session: SessionRecord,
    rec: Recording,
    wp: WindowParams,
    rng: random.Random,
) -> tuple[list[WindowRecord], WindowingStats, tuple, tuple[tuple[float, float], ...]]:
    """build_windows + exclusion handling per ``wp.exclusion_mode``.

    Returns ``(records, stats, turns_for_chunk_task, spans_for_chunk_task)``.
    "drop" mode (and any session without spans) is the exact original code:
    one build_windows call over the whole session, then windows overlapping a
    span are discarded.  "cut" mode plans each segment from ``cut_segments``
    in order with the SAME rng object, so the draw sequence is deterministic.
    """
    kwargs = dict(
        window_min=wp.window_min,
        window_max=wp.window_max,
        boundary_guard=wp.boundary_guard,
        tail_min=wp.tail_min,
        rng=rng,
        trim_to_turns=wp.trim_to_turns,
        min_coverage=wp.min_coverage,
        snap_start_to_turn=wp.snap_start_to_turn,
    )
    if wp.exclusion_mode == "cut" and session.exclusion_spans:
        segments, spans, n_excluded = cut_segments(session)
        records: list[WindowRecord] = []
        stats = WindowingStats()
        stats.excluded_sec = sum(b - a for a, b in spans)
        stats.n_excluded_turns = n_excluded
        kept_turns: list = []
        for a, b, turns in segments:
            kept_turns.extend(turns)
            if not turns:
                continue
            seg_records, seg_stats = build_windows(
                session.session_id,
                rec,
                turns,
                region=(a, b),
                first_index=len(records),
                **kwargs,
            )
            records.extend(seg_records)
            stats.merge(seg_stats)
        return records, stats, tuple(kept_turns), spans
    records, stats = build_windows(session.session_id, rec, session.turns, **kwargs)
    if session.exclusion_spans:
        surviving = []
        for record in records:
            if any(record.t0 < b and a < record.t1 for a, b in session.exclusion_spans):
                stats.n_windows -= 1
            else:
                surviving.append(record)
        records = surviving
    return records, stats, session.turns, session.exclusion_spans


def plan_session(
    session: SessionRecord,
    *,
    params: WindowParams,
    seed: int,
    epoch: int | None,
    chunk_params: ChunkTaskParams | None = None,
    timestamp_align_prob: float = 0.0,
    context_channel_prob: float = 0.0,
    independent_mask_prob: float = 0.0,
) -> tuple[list[WindowRecord], WindowingStats]:
    if session.atomic:
        stats = WindowingStats()
        stats.n_windows = 1
        records = [
            WindowRecord(
                window_id=session.window_id or f"{session.session_id}_w00000",
                session_id=session.session_id,
                audio_relpath=session.audio_relpath,
                num_channels=session.num_channels,
                sample_rate=session.sample_rate,
                t0=0.0,
                t1=session.duration,
                turns=session.turns,
            )
        ]
        records = _apply_timestamp_coin(
            records, stats, session, seed, epoch, timestamp_align_prob
        )
        records = _apply_mask_coin(
            records,
            stats,
            session,
            seed,
            epoch,
            context_channel_prob,
            independent_mask_prob,
        )
        return records, stats

    rec = Recording(
        id=session.session_id,
        audio_relpath=session.audio_relpath,
        sample_rate=session.sample_rate,
        num_channels=session.num_channels,
        duration=session.duration,
    )

    # BIT-PARITY: chunk_params is None / chunk_task_prob == 0 / epoch is None
    # all take the exact pre-chunk-task code path below, with the exact same
    # RNG construction and zero extra draws - required for the golden parity
    # tests and for valid/test/inference (epoch=None), which must never chunk.
    chunking = (
        chunk_params is not None
        and chunk_params.chunk_task_prob > 0
        and epoch is not None
    )

    if not chunking:
        records, stats, _turns, _spans = _build_session_windows(
            session, rec, params, _window_rng(seed, session.session_id, epoch)
        )
        records = _apply_timestamp_coin(
            records, stats, session, seed, epoch, timestamp_align_prob
        )
        records = _apply_mask_coin(
            records,
            stats,
            session,
            seed,
            epoch,
            context_channel_prob,
            independent_mask_prob,
        )
        return records, stats

    # Chunk-task mode: one rng constructed once, the per-session coin drawn
    # FIRST from it, then the same rng object flows to build_windows and to
    # every draw_chunk_task call below (fixed draw order == determinism).
    rng = _window_rng(seed, session.session_id, epoch)
    is_chunk = rng.random() < chunk_params.chunk_task_prob
    window_params = params
    if is_chunk:
        window_params = dataclasses.replace(
            params,
            window_min=chunk_params.chunk_window_min,
            window_max=chunk_params.chunk_window_max,
        )
    # Exclusion-span handling (drop or cut, per params) runs BEFORE the
    # draw_chunk_task calls: draws only happen per SURVIVING window, so no
    # draws are wasted on dropped windows and the surviving set alone
    # determines the draw sequence.
    records, stats, chunk_turns, chunk_spans = _build_session_windows(
        session, rec, window_params, rng
    )

    if is_chunk:
        chunked_records = []
        for record in records:
            plan, degraded = draw_chunk_task(
                record,
                chunk_turns,
                session.duration,
                chunk_spans,
                session.num_channels,
                rng,
                chunk_params,
            )
            if plan is None:
                stats.n_chunk_fallback_infill += 1
                chunked_records.append(record)
                continue
            if plan.kind == "full":
                stats.n_chunk_full += 1
            else:
                stats.n_chunk_prompt_only += 1
            if degraded:
                stats.n_chunk_degraded += 1
            chunked_records.append(dataclasses.replace(record, chunk_task=plan))
        records = chunked_records

    records = _apply_timestamp_coin(
        records, stats, session, seed, epoch, timestamp_align_prob
    )
    records = _apply_mask_coin(
        records,
        stats,
        session,
        seed,
        epoch,
        context_channel_prob,
        independent_mask_prob,
    )
    return records, stats


def plan_sessions(
    sessions: Iterable[SessionRecord],
    *,
    params: WindowParams,
    seed: int,
    epoch: int | None,
    chunk_params: ChunkTaskParams | None = None,
    timestamp_align_prob: float = 0.0,
    context_channel_prob: float = 0.0,
    independent_mask_prob: float = 0.0,
) -> tuple[list[WindowRecord], WindowingStats]:
    all_records: list[WindowRecord] = []
    total = WindowingStats()
    for session in sessions:
        records, stats = plan_session(
            session,
            params=params,
            seed=seed,
            epoch=epoch,
            chunk_params=chunk_params,
            timestamp_align_prob=timestamp_align_prob,
            context_channel_prob=context_channel_prob,
            independent_mask_prob=independent_mask_prob,
        )
        all_records.extend(records)
        total.merge(stats)
    return all_records, total
