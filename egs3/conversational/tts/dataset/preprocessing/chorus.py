"""NSF Chorus corpus parsing (multi-party meetings, 4-8 close-talk channels).

Layout (verified on Delta 2026-08-27): ``<root>/manifest.jsonl`` with one
meeting per line, and one mono 16-bit 24 kHz wav per speaker at
``<root>/<split>/<MTG_id>/<Name>.wav``; all wavs of a meeting share one
frame count.  Transcripts carry inline markup (``<ST/>`` sentence marks,
``<FILL/>``, ``<PName>x</PName>`` ...) that ``clean_chorus_text`` strips.

Channel convention: channel i of a meeting is its i-th speaker in SORTED
name order; ``merge_all`` joins the wavs in that order so the manifest's
channel ids and the merged FLAC's channels agree by construction.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

from .fisher import CleanResult
from .sssd import Supervision

CHORUS_SAMPLE_RATE = 24000

# Self-closing tags that carry no words (sentence marks, fillers, laughter,
# backchannels, annotator issues, false starts, pauses, sniffs).
KNOWN_TAGS: frozenset[str] = frozenset(
    {
        "<ST/>",
        "<FILL/>",
        "<FILLlaugh/>",
        "<BA/>",
        "<ISSUE/>",
        "<FL/>",
        "<PAUSE/>",
        "<SN/>",
    }
)
UNKNOWN_TAG = "<UNKNOWN/>"
_PNAME_RE = re.compile(r"<PName>(.*?)</PName>")
_ANY_TAG_RE = re.compile(r"<[^<>]+>")


@dataclass(frozen=True)
class ChorusMeeting:
    meeting_id: str
    split: str
    duration: float  # manifest value; the builder re-measures from FLAC headers
    sample_rate: int
    speakers: tuple[str, ...]  # sorted; index == channel
    wavs: tuple[str, ...]  # root-relative, aligned with speakers
    utterances: tuple[Supervision, ...]

    @property
    def num_channels(self) -> int:
        return len(self.speakers)


def load_chorus_manifest(path: str | Path) -> dict[str, ChorusMeeting]:
    """Parse ``manifest.jsonl`` into meetings keyed by ``meeting_id``.

    Args:
        path: The corpus ``manifest.jsonl``.

    Returns:
        ``{meeting_id: ChorusMeeting}`` with speakers sorted by name and
        every utterance's ``channel`` set to its speaker's sorted index.

    Raises:
        ValueError: on a duplicate meeting id, a sample rate other than
            24000, or a speaker count that disagrees with ``n_speakers``.
    """
    meetings: dict[str, ChorusMeeting] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            mid = d["meeting_id"]
            if mid in meetings:
                raise ValueError(f"duplicate meeting id {mid!r} in {path}")
            sr = int(d["sr"])
            if sr != CHORUS_SAMPLE_RATE:
                raise ValueError(
                    f"meeting {mid!r}: sr {sr} != {CHORUS_SAMPLE_RATE} "
                    "(the Chorus wavs are 24 kHz)"
                )
            names = tuple(sorted(d["speakers"]))
            if len(names) != int(d["n_speakers"]):
                raise ValueError(
                    f"meeting {mid!r}: {len(names)} speakers listed, "
                    f"n_speakers={d['n_speakers']}"
                )
            wavs = tuple(d["speakers"][n]["wav"] for n in names)
            utts: list[Supervision] = []
            for ch, name in enumerate(names):
                for k, (start, end, text) in enumerate(
                    d["speakers"][name]["utterances"]
                ):
                    utts.append(
                        Supervision(
                            id=f"{mid}-{name}-{k:04d}",
                            recording_id=mid,
                            channel=ch,
                            start=float(start),
                            duration=float(end) - float(start),
                            text=str(text),
                            speaker=name,
                        )
                    )
            utts.sort(key=lambda s: (s.start, s.channel))
            meetings[mid] = ChorusMeeting(
                meeting_id=mid,
                split=str(d["split"]),
                duration=float(d["duration"]),
                sample_rate=sr,
                speakers=names,
                wavs=wavs,
                utterances=tuple(utts),
            )
    return meetings


def clean_chorus_text(text: str) -> CleanResult:
    """Strip Chorus markup from one utterance (see module docstring).

    ``<PName>x</PName>`` unwraps to ``x``; known self-closing tags are
    deleted.  ``<UNKNOWN/>`` marks a spoken word the transcriber could not
    make out: any utterance containing one (inline or alone) is
    UNINTELLIGIBLE under Fisher's standard - its audio holds words its text
    would not cover - so it is dropped and its span excluded.  The planner's
    ``exclusion_mode: cut`` keeps that from costing whole windows (design
    2026-08-28).  Any other tag is an error so corpus changes surface at
    build time.

    Args:
        text: Raw utterance text.

    Returns:
        ``CleanResult(text, unintelligible)``.

    Raises:
        ValueError: on a tag outside ``KNOWN_TAGS`` / ``<UNKNOWN/>`` / PName.
    """
    cleaned = _PNAME_RE.sub(r"\1", text)
    for tag in _ANY_TAG_RE.findall(cleaned):
        if tag not in KNOWN_TAGS and tag != UNKNOWN_TAG:
            raise ValueError(f"unknown Chorus markup {tag} in {text!r}")
    if UNKNOWN_TAG in text:
        return CleanResult("", True)
    cleaned = _ANY_TAG_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    return CleanResult(cleaned, False)


def clean_chorus_supervisions(
    sups: Sequence[Supervision],
) -> tuple[list[Supervision], list[tuple[float, float]], int]:
    """Apply ``clean_chorus_text`` per utterance BEFORE turn merging.

    Args:
        sups: One meeting's utterances.

    Returns:
        ``(kept, unintelligible_spans, n_benign_dropped)`` with the same
        contract as ``fisher.clean_fisher_supervisions``: the spans become
        the session's ``exclusion_spans``.
    """
    kept: list[Supervision] = []
    spans: list[tuple[float, float]] = []
    n_benign = 0
    for s in sups:
        res = clean_chorus_text(s.text)
        if res.unintelligible:
            spans.append((s.start, s.start + s.duration))
        elif not res.text:
            n_benign += 1
        else:
            kept.append(dataclasses.replace(s, text=res.text))
    return kept, spans, n_benign


def merged_relpath(m: ChorusMeeting) -> str:
    """Merged N-channel FLAC path relative to the flac dir: ``<split>/<id>.flac``."""
    return f"{m.split}/{m.meeting_id}.flac"


def _merge_one(job: tuple[str, list[str], str, int, str]) -> tuple[str, bool]:
    """(mid, source wavs in channel order, flac_path, expected_frames, ffmpeg)
    -> (mid, newly_written).  Atomic: PID-unique tmp, header check
    (channels == len(sources), frames == expected), ``os.replace``."""
    mid, sources, flac_path, expected_frames, ffmpeg = job
    flac = Path(flac_path)
    if flac.is_file():
        return mid, False
    for src in sources:
        if not Path(src).is_file():
            raise FileNotFoundError(f"Chorus source audio not found: {src}")
    n = len(sources)
    tmp = flac.with_name(f"{flac.name}.{os.getpid()}.tmp")
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    for src in sources:
        cmd += ["-i", str(src)]
    inputs = "".join(f"[{i}:a]" for i in range(n))
    cmd += [
        "-filter_complex",
        f"{inputs}amerge=inputs={n}[out]",
        "-map",
        "[out]",
        "-c:a",
        "flac",
        "-f",
        "flac",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    import soundfile as sf

    info = sf.info(str(tmp))
    if info.channels != n or info.frames != expected_frames:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{flac.name}: merged file has {info.channels} channels / "
            f"{info.frames} frames, expected {n} / {expected_frames}"
        )
    os.replace(tmp, flac)
    return mid, True


def merge_all(
    meetings: dict[str, ChorusMeeting],
    corpus_root: str | Path,
    flac_dir: str | Path,
    ffmpeg: str = "ffmpeg",
    workers: int = 4,
) -> int:
    """Idempotent parallel per-speaker wav -> N-channel FLAC merge.

    The expected frame count comes from the first source wav's header (all
    wavs of a meeting share one frame count); the manifest ``duration`` is
    informational only.

    Args:
        meetings: Output of ``load_chorus_manifest``.
        corpus_root: Directory holding ``<split>/<id>/<Name>.wav``.
        flac_dir: Target dir for ``<split>/<id>.flac``.
        ffmpeg: ffmpeg executable.
        workers: Process count; ``<= 1`` runs serially in-process.

    Returns:
        Number of FLACs newly written.

    Raises:
        FileNotFoundError: on a missing source wav.
        RuntimeError: when a merged file's header disagrees with the sources.
    """
    import soundfile as sf

    flac_dir = Path(flac_dir)
    root = Path(corpus_root)
    jobs = []
    for mid, m in sorted(meetings.items()):
        final = flac_dir / merged_relpath(m)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.is_file():
            continue
        sources = [str(root / w) for w in m.wavs]
        first = Path(sources[0])
        if not first.is_file():
            raise FileNotFoundError(f"Chorus source audio not found: {first}")
        expected = sf.info(str(first)).frames
        jobs.append((mid, sources, str(final), expected, ffmpeg))
    # Any ``*.tmp`` here is garbage abandoned by a killed prior run.
    for stale in flac_dir.rglob("*.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass
    if not jobs:
        return 0
    written = 0
    if int(workers) <= 1:
        for job in jobs:
            written += int(_merge_one(job)[1])
        return written
    with Pool(processes=int(workers)) as pool:
        for _mid, did_write in pool.imap_unordered(_merge_one, jobs):
            written += int(did_write)
    return written


def measured_durations_nch(
    meetings: dict[str, ChorusMeeting], flac_dir: str | Path
) -> dict[str, float]:
    """Decoded duration per meeting from the merged FLAC headers.

    Args:
        meetings: Output of ``load_chorus_manifest``.
        flac_dir: Directory holding the merged FLACs.

    Returns:
        ``{meeting_id: seconds}``.

    Raises:
        RuntimeError: on a rate or channel-count mismatch with the manifest.
    """
    import soundfile as sf

    out: dict[str, float] = {}
    for mid, m in meetings.items():
        info = sf.info(str(Path(flac_dir) / merged_relpath(m)))
        if info.samplerate != m.sample_rate or info.channels != m.num_channels:
            raise RuntimeError(
                f"{merged_relpath(m)}: {info.samplerate} Hz / {info.channels} "
                f"channels, manifest says {m.sample_rate} Hz / "
                f"{m.num_channels} channels"
            )
        out[mid] = info.frames / info.samplerate
    return out
