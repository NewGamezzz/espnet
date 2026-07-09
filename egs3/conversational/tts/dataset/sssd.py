"""SSSD corpus ingestion: lhotse-manifest parsing, path remapping, turn merging.

Parses ``recordings.jsonl.gz`` / ``supervisions.jsonl.gz`` with plain
``gzip`` + ``json`` (no lhotse dependency).  Absolute audio paths inside the
manifests are valid only on the machine that wrote them, so recordings keep a
path *relative* to the dataset root and callers join it themselves.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class Supervision:
    id: str
    recording_id: str
    channel: int
    start: float
    duration: float
    text: str
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Recording:
    id: str
    audio_relpath: str  # relative to dataset_root, e.g. "original/<id>_mixed.flac"
    sample_rate: int
    num_channels: int
    duration: float


@dataclass(frozen=True)
class Turn:
    channel: int
    speaker: str
    text: str
    start: float
    end: float


def _iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_recordings(path: str | Path, audio_subdir: str = "original") -> dict[str, Recording]:
    """Parse ``recordings.jsonl.gz`` into ``Recording`` objects keyed by id.

    The absolute prefix of each source path is discarded and replaced by
    ``<audio_subdir>/<basename>``; only the dataset-root-relative location is
    trusted across machines.
    """
    recordings: dict[str, Recording] = {}
    for rec in _iter_jsonl_gz(Path(path)):
        sources = rec["sources"]
        if len(sources) != 1:
            raise ValueError(
                f"recording {rec['id']!r} has {len(sources)} sources; this pipeline "
                "assumes one multi-channel file per session (channel order would be "
                "ambiguous otherwise)"
            )
        channel_ids = rec.get("channel_ids") or sources[0]["channels"]
        recordings[rec["id"]] = Recording(
            id=rec["id"],
            audio_relpath=f"{audio_subdir}/{Path(sources[0]['source']).name}",
            sample_rate=int(rec["sampling_rate"]),
            num_channels=len(channel_ids),
            duration=float(rec["duration"]),
        )
    return recordings


def _parse_channel(raw) -> int:
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(f"expected a single channel, got {raw}")
        return int(raw[0])
    return int(raw)


def load_supervisions(
    path: str | Path, recordings: dict[str, Recording] | None = None
) -> dict[str, list[Supervision]]:
    """Parse ``supervisions.jsonl.gz`` keyed by recording id, sorted (start, channel).

    When ``recordings`` is given, supervision spans are clamped to the
    recording duration and entries for unknown recordings raise.
    """
    per_recording: dict[str, list[Supervision]] = {}
    for sup in _iter_jsonl_gz(Path(path)):
        recording_id = sup["recording_id"]
        start = float(sup["start"])
        duration = float(sup["duration"])
        if start < 0:
            raise ValueError(f"supervision {sup['id']!r} has negative start {start}")
        if recordings is not None:
            if recording_id not in recordings:
                raise ValueError(
                    f"supervision {sup['id']!r} references unknown recording "
                    f"{recording_id!r}"
                )
            duration = min(duration, recordings[recording_id].duration - start)
        per_recording.setdefault(recording_id, []).append(
            Supervision(
                id=sup["id"],
                recording_id=recording_id,
                channel=_parse_channel(sup["channel"]),
                start=start,
                duration=duration,
                text=sup.get("text") or "",
                speaker=sup.get("speaker") or "",
            )
        )
    for sups in per_recording.values():
        sups.sort(key=lambda s: (s.start, s.channel))
    return per_recording


def merge_turns(sups: Sequence[Supervision], merge_gap: float) -> list[Turn]:
    """Merge consecutive same-channel utterances into turns.

    Per channel (sorted by start), an utterance merges into the running turn
    when the gap to it is below ``merge_gap``; negative gaps (overlapping ASR
    spans) merge too.  Merged text joins with single spaces.  Merging is
    per-channel-stream and gap-based only: an interleaved backchannel on
    another channel does not split a turn.  The returned global turn order is
    sorted by (start, channel).
    """
    per_channel: dict[int, list[Supervision]] = {}
    for sup in sups:
        per_channel.setdefault(sup.channel, []).append(sup)

    turns: list[Turn] = []
    for channel, channel_sups in per_channel.items():
        channel_sups = sorted(channel_sups, key=lambda s: s.start)
        group: list[Supervision] = []
        for sup in channel_sups:
            if group and sup.start - max(s.end for s in group) < merge_gap:
                group.append(sup)
            else:
                if group:
                    turns.append(_group_to_turn(channel, group))
                group = [sup]
        if group:
            turns.append(_group_to_turn(channel, group))

    turns.sort(key=lambda t: (t.start, t.channel))
    return turns


def _group_to_turn(channel: int, group: list[Supervision]) -> Turn:
    return Turn(
        channel=channel,
        speaker=group[0].speaker,
        text=" ".join(s.text.strip() for s in group if s.text.strip()),
        start=group[0].start,
        end=max(s.end for s in group),
    )


def occupied_intervals(turns: Sequence[Turn]) -> list[tuple[float, float]]:
    """Merged union of turn spans across all channels, sorted."""
    spans = sorted((t.start, t.end) for t in turns)
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def session_speakers(sups: Sequence[Supervision]) -> frozenset[str]:
    return frozenset(s.speaker for s in sups if s.speaker)
