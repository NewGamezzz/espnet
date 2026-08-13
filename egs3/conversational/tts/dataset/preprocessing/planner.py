"""Per-epoch window planning over session records (the online chunker).

Pure module: wraps windows.build_windows without modifying it.  Frozen mode
(epoch=None) seeds random.Random(f"{seed}:window:{sid}") - the exact stream
the retired offline builders used - so it reproduces their window manifests
bit-for-bit; that mode serves valid/test splits, inference, and the golden
parity tests.  Epoch mode appends ":epoch{epoch}" for fresh training windows
each epoch that any rank or worker can re-derive independently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .sessions import SessionRecord
from .sssd import Recording
from .windows import WindowingStats, WindowRecord, build_windows


@dataclass(frozen=True)
class WindowParams:
    """Training-time windowing knobs (ratified 2026-08-06 live values)."""

    window_min: float = 10.0
    window_max: float = 80.0
    boundary_guard: float = 0.0
    tail_min: float = 5.0
    trim_to_turns: bool = True
    min_coverage: float = 0.0
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
            if any(record.t0 < b and a < record.t1 for a, b in session.exclusion_spans):
                stats.n_windows -= 1
            else:
                surviving.append(record)
        records = surviving
    return records, stats


def plan_sessions(
    sessions: Iterable[SessionRecord],
    *,
    params: WindowParams,
    seed: int,
    epoch: int | None,
) -> tuple[list[WindowRecord], WindowingStats]:
    all_records: list[WindowRecord] = []
    total = WindowingStats()
    for session in sessions:
        records, stats = plan_session(session, params=params, seed=seed, epoch=epoch)
        all_records.extend(records)
        total.merge(stats)
    return all_records, total
