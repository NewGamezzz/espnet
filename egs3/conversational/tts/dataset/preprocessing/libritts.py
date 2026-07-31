"""LibriTTS corpus ingestion: subset scanning and utterance-as-window records.

A LibriTTS utterance becomes a 1-channel ``WindowRecord`` (one turn on
channel 0 spanning the whole file), so everything downstream of the manifest
(dataset, preprocessor, collator, sampler, model) is reused unchanged.
Corpus-specific code stays in this module per the generalization contract in
the design note: adding another corpus means another module like this one,
never changes downstream of the manifest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .sssd import Turn
from .windows import WindowRecord


@dataclass(frozen=True)
class UttEntry:
    """One scanned utterance; ``text`` is the raw transcript (normalization
    against the extended-vocab charset happens in the builder)."""

    utt_id: str
    audio_relpath: str  # relative to the LibriTTS root
    speaker: str
    chapter: str
    text: str


def scan_subset(root: Path, subset: str) -> list[UttEntry]:
    """Scan ``<root>/<subset>`` for ``*.normalized.txt`` files with a sibling
    ``.wav``; skips orphans and empty transcripts.  Sorted by transcript path,
    so output order is deterministic across filesystems."""
    subset_dir = Path(root) / subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"LibriTTS subset not found: {subset_dir}")
    entries = []
    for text_path in sorted(subset_dir.rglob("*.normalized.txt")):
        wav_path = text_path.with_name(
            text_path.name.replace(".normalized.txt", ".wav")
        )
        if not wav_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        entries.append(
            UttEntry(
                utt_id=wav_path.stem,
                audio_relpath=str(wav_path.relative_to(root)),
                speaker=text_path.parent.parent.name,
                chapter=text_path.parent.name,
                text=text,
            )
        )
    return entries


def utterance_record(
    entry: UttEntry, duration: float, sample_rate: int, text: str
) -> WindowRecord:
    """One N=1 window spanning the whole utterance file.

    ``text`` is the NORMALIZED transcript (the builder normalizes against the
    extended-vocab charset, mirroring the SSSD build so ``<OTHER>`` counts and
    token coverage can never diverge between corpora).
    """
    return WindowRecord(
        window_id=f"libritts_{entry.utt_id}",
        session_id=f"libritts_{entry.speaker}_{entry.chapter}",
        audio_relpath=entry.audio_relpath,
        num_channels=1,
        sample_rate=sample_rate,
        t0=0.0,
        t1=duration,
        turns=(
            Turn(
                channel=0,
                speaker=entry.speaker,
                text=text,
                start=0.0,
                end=duration,
            ),
        ),
    )


def subsample_to_hours(
    items: Sequence[tuple[UttEntry, float]], hours: float, seed: int
) -> list[tuple[UttEntry, float]]:
    """Seeded shuffle, then greedy prefix until the duration budget is met.

    Overshoots by at most one utterance; the result is re-sorted by utt_id so
    the manifest order is stable regardless of the shuffle."""
    if hours <= 0:
        raise ValueError(f"hours must be positive, got {hours}")
    pool = list(items)
    random.Random(f"{seed}:libritts_valid").shuffle(pool)
    budget = hours * 3600.0
    taken: list[tuple[UttEntry, float]] = []
    total = 0.0
    for entry, dur in pool:
        if total >= budget:
            break
        taken.append((entry, dur))
        total += dur
    return sorted(taken, key=lambda item: item[0].utt_id)
