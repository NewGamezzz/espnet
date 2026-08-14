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
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Iterable

from .chunk_task import ChunkTaskParams, draw_chunk_task
from .sessions import SessionRecord
from .sssd import Recording
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


def _window_rng(seed: int, session_id: str, epoch: int | None) -> random.Random:
    if epoch is None:
        return random.Random(f"{seed}:window:{session_id}")
    return random.Random(f"{seed}:window:{session_id}:epoch{epoch}")


def plan_session(
    session: SessionRecord,
    *,
    params: WindowParams,
    seed: int,
    epoch: int | None,
    chunk_params: ChunkTaskParams | None = None,
) -> tuple[list[WindowRecord], WindowingStats]:
    if session.atomic:
        stats = WindowingStats()
        stats.n_windows = 1
        return [
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
        ], stats

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
        records, stats = build_windows(
            session.session_id,
            rec,
            session.turns,
            window_min=params.window_min,
            window_max=params.window_max,
            boundary_guard=params.boundary_guard,
            tail_min=params.tail_min,
            rng=_window_rng(seed, session.session_id, epoch),
            trim_to_turns=params.trim_to_turns,
            min_coverage=params.min_coverage,
            snap_start_to_turn=params.snap_start_to_turn,
        )
        if session.exclusion_spans:
            surviving = []
            for record in records:
                if any(
                    record.t0 < b and a < record.t1 for a, b in session.exclusion_spans
                ):
                    stats.n_windows -= 1
                else:
                    surviving.append(record)
            records = surviving
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
    records, stats = build_windows(
        session.session_id,
        rec,
        session.turns,
        window_min=window_params.window_min,
        window_max=window_params.window_max,
        boundary_guard=window_params.boundary_guard,
        tail_min=window_params.tail_min,
        rng=rng,
        trim_to_turns=window_params.trim_to_turns,
        min_coverage=window_params.min_coverage,
        snap_start_to_turn=window_params.snap_start_to_turn,
    )
    # Exclusion-span filtering runs BEFORE draw_chunk_task calls: draws only
    # happen per SURVIVING window, so no draws are wasted on dropped windows
    # and the surviving set alone determines the draw sequence.
    if session.exclusion_spans:
        surviving = []
        for record in records:
            if any(record.t0 < b and a < record.t1 for a, b in session.exclusion_spans):
                stats.n_windows -= 1
            else:
                surviving.append(record)
        records = surviving

    if is_chunk:
        chunked_records = []
        for record in records:
            plan, degraded = draw_chunk_task(
                record,
                session.turns,
                session.duration,
                session.exclusion_spans,
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

    return records, stats


def plan_sessions(
    sessions: Iterable[SessionRecord],
    *,
    params: WindowParams,
    seed: int,
    epoch: int | None,
    chunk_params: ChunkTaskParams | None = None,
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
        )
        all_records.extend(records)
        total.merge(stats)
    return all_records, total
