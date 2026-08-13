"""Session-level records: the build-stage output the online planner consumes.

One record per session (or per utterance for atomic corpora like LibriTTS)
carrying the merged + NORMALIZED turns.  Normalization happens once at build
time (see preprocessing/text.py: <OTHER> counts must never desync between
branches); window planning happens online (preprocessing/planner.py).

Floats are serialized UNROUNDED: json round-trips Python floats exactly, and
rounding turn times here would perturb build_windows inputs by < 1e-6,
breaking bit-parity between the frozen planner and the retired offline
manifests.  Only windows.to_json rounds, exactly as before.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .sssd import Turn


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    audio_relpath: str  # relative to the runtime dataset_root
    num_channels: int
    sample_rate: int  # source rate, not the training rate
    duration: float  # measured audio duration in seconds
    turns: tuple[Turn, ...]  # merged + normalized, absolute session seconds
    # Windows overlapping any (start, end) span are dropped by the planner
    # (Fisher's unintelligible spans; empty for other corpora).
    exclusion_spans: tuple[tuple[float, float], ...] = ()
    # Atomic records (LibriTTS utterances) bypass planning: one window
    # spanning the whole file, with window_id preserved verbatim.
    atomic: bool = False
    window_id: str | None = None


def to_json(s: SessionRecord) -> dict:
    return {
        "session_id": s.session_id,
        "audio_relpath": s.audio_relpath,
        "num_channels": s.num_channels,
        "sample_rate": s.sample_rate,
        "duration": s.duration,
        "atomic": s.atomic,
        "window_id": s.window_id,
        "exclusion_spans": [list(span) for span in s.exclusion_spans],
        "turns": [
            {
                "channel": t.channel,
                "speaker": t.speaker,
                "text": t.text,
                "start": t.start,
                "end": t.end,
            }
            for t in s.turns
        ],
    }


def from_json(d: dict) -> SessionRecord:
    return SessionRecord(
        session_id=d["session_id"],
        audio_relpath=d["audio_relpath"],
        num_channels=int(d["num_channels"]),
        sample_rate=int(d["sample_rate"]),
        duration=float(d["duration"]),
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
        exclusion_spans=tuple(
            (float(a), float(b)) for a, b in d.get("exclusion_spans", [])
        ),
        atomic=bool(d.get("atomic", False)),
        window_id=d.get("window_id"),
    )


def write_session_manifest(path, records) -> int:
    """One JSON object per line; atomic .tmp + os.replace like the window
    manifests were (a build killed mid-write must never leave a truncated
    file that existence-only is_built treats as built)."""
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


def read_session_manifest(path) -> list[SessionRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(from_json(json.loads(line)))
    if not records:
        raise RuntimeError(f"Session manifest is empty: {path}")
    return records
