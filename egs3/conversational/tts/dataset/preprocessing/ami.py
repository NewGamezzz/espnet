"""AMI corpus ingestion for the beyond-two-speakers evaluation.

The AMI test partition (24 meetings, four participants each) ships as one
16 kHz headset wav per participant plus NXT word annotations with forced-
aligned ``starttime``/``endtime`` per word.  This module parses the two XML
resources the recipe needs (``corpusResources/meetings.xml`` for the agent ->
headset channel -> global speaker mapping, ``words/<MID>.<A-D>.words.xml``
for timed words), turns word runs into ``Supervision`` rows, and transcodes
the four headset wavs into ONE 4-channel 24 kHz FLAC per meeting so the
shared window loader sees the same shape as SSSD / Fisher sessions.

Design note: "Design - Beyond Two Speakers Evaluation on AMI", section 1.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .sssd import Recording, Supervision

NITE_NS = "{http://nite.sourceforge.net/}"

# Full-corpus partition, test set (groups.inf.ed.ac.uk/ami/corpus/datasets.shtml).
_TEST_SERIES = ("EN2002", "ES2004", "ES2014", "IS1009", "TS3003", "TS3007")
TEST_MEETINGS: tuple[str, ...] = tuple(
    f"{series}{session}" for series in _TEST_SERIES for session in "abcd"
)

SOURCE_SAMPLE_RATE = 16000
TARGET_SAMPLE_RATE = 24000
NUM_HEADSETS = 4
AGENTS = "ABCD"


@dataclass(frozen=True)
class Participant:
    agent: str  # NXT agent letter A-D (the words file suffix)
    channel: int  # headset index 0-3 (the Headset-<ch>.wav suffix)
    global_name: str  # corpus-wide speaker id, e.g. FEE013


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    punc: bool


def validate_participants(mid: str, parts: Sequence[Participant]) -> None:
    """A usable meeting has 1 to 4 distinct speakers on distinct headset
    channels within 0..3 (EN2002c, a test meeting, has three: channels
    1-3); the window loader would otherwise mis-assign audio rows silently.
    An absent headset channel carries no supervisions and never becomes an
    active channel."""
    channels = sorted(p.channel for p in parts)
    names = {p.global_name for p in parts}
    if (
        not 1 <= len(parts) <= NUM_HEADSETS
        or len(set(channels)) != len(parts)
        or any(not 0 <= c < NUM_HEADSETS for c in channels)
        or len(names) != len(parts)
    ):
        raise ValueError(
            f"{mid}: expected 1..{NUM_HEADSETS} distinct speakers on distinct "
            f"channels within 0..{NUM_HEADSETS - 1}, got channels {channels} "
            f"names {sorted(names)}"
        )


def load_meetings(
    path: str | Path, require: Sequence[str] | None = None
) -> dict[str, list[Participant]]:
    """Parse ``corpusResources/meetings.xml`` into ``{meeting_id: participants}``.

    The file covers the whole corpus, and some non-test meetings (IN1001,
    for one) have three participants, so validation runs only on
    ``require`` - the meetings the caller will use - each of which must be
    present and pass :func:`validate_participants`.
    """
    root = ET.parse(str(path)).getroot()
    out: dict[str, list[Participant]] = {}
    for meeting in root.iter("meeting"):
        mid = meeting.get("observation")
        if not mid:
            continue
        out[mid] = [
            Participant(
                agent=spk.get("nxt_agent"),
                channel=int(spk.get("channel")),
                global_name=spk.get("global_name"),
            )
            for spk in meeting.iter("speaker")
        ]
    for mid in require or ():
        if mid not in out:
            raise KeyError(f"{mid}: not in {path}")
        validate_participants(mid, out[mid])
    return out


def load_words(path: str | Path) -> list[Word]:
    """Parse one ``<MID>.<agent>.words.xml``.

    Keeps ``<w>`` elements that carry both times; ``<vocalsound>``,
    ``<nonvocalsound>``, ``<gap>`` and untimed words are dropped.  Word
    start times must be non-decreasing (forced alignment output is),
    otherwise the file is corrupt for our purposes.
    """
    root = ET.parse(str(path)).getroot()
    words: list[Word] = []
    for el in root:
        if el.tag != "w":
            continue
        start, end = el.get("starttime"), el.get("endtime")
        if start is None or end is None:
            continue
        words.append(
            Word(
                start=float(start),
                end=float(end),
                text=(el.text or "").strip(),
                punc=el.get("punc") == "true",
            )
        )
    for a, b in zip(words, words[1:]):
        if b.start + 1e-6 < a.start:
            raise ValueError(
                f"{path}: word timings are not monotone at {b.start} < {a.start}"
            )
    return words


_PUNCT_RE = re.compile(r"\s+([.,?!;:])")
_SPACES_RE = re.compile(r"\s+")


def normalize_ami_text(text: str) -> str:
    """AMI-specific text cleanup, applied BEFORE the shared vocab normalizer.

    * spelled acronyms ``L_C_D`` -> ``L C D`` (the underscore is AMI's
      letter separator, and F5 reads single capitals as letters);
    * punctuation tokens attach to the preceding word;
    * ALL-CAPS words (2+ letters) are lowercased - F5 spells capitalised
      runs out as initialisms (the CoVoMix2 3-speaker v1 lesson);
    * whitespace collapsed.
    """
    tokens: list[str] = []
    for tok in text.split():
        parts = tok.split("_")
        if len(parts) > 1 and all(len(p) == 1 and p.isalpha() for p in parts):
            tokens.extend(parts)
            continue
        if len(tok) >= 2 and tok.isalpha() and tok.isupper():
            tok = tok.lower()
        tokens.append(tok)
    joined = " ".join(tokens)
    joined = _PUNCT_RE.sub(r"\1", joined)
    return _SPACES_RE.sub(" ", joined).strip()


def words_to_supervisions(
    words: Sequence[Word],
    *,
    meeting_id: str,
    channel: int,
    speaker: str,
    utterance_gap: float,
    agent: str | None = None,
) -> list[Supervision]:
    """Group consecutive words into utterances split at silences longer than
    ``utterance_gap`` seconds; punctuation never opens an utterance and a run
    with no alphanumeric character is dropped.  Ids are
    ``<MID>.<agent>.u<NNNN>``; ``agent`` defaults to the A-D letter of the
    channel, which is the corpus convention (verified on meetings.xml)."""
    runs: list[list[Word]] = []
    cur: list[Word] = []
    last_end: float | None = None
    for w in words:
        if w.punc and not cur:
            continue
        if cur and last_end is not None and w.start - last_end > utterance_gap:
            runs.append(cur)
            cur = []
        cur.append(w)
        last_end = w.end if last_end is None else max(last_end, w.end)
    if cur:
        runs.append(cur)

    agent = agent if agent is not None else AGENTS[channel]
    sups: list[Supervision] = []
    for run in runs:
        text = normalize_ami_text(" ".join(w.text for w in run))
        if not any(c.isalnum() for c in text):
            continue
        start = run[0].start
        end = max(w.end for w in run)
        sups.append(
            Supervision(
                id=f"{meeting_id}.{agent}.u{len(sups):04d}",
                recording_id=meeting_id,
                channel=channel,
                start=round(start, 6),
                duration=round(end - start, 6),
                text=text,
                speaker=speaker,
            )
        )
    return sups


def complete_participants(
    mid: str, parts: Sequence[Participant], words_dir: str | Path
) -> tuple[list[Participant], list[Participant]]:
    """Add agents that have a timed words file but no roster entry.

    ``meetings.xml`` is incomplete for EN2002c: agent A has 3,716 timed
    words on headset 0 but no ``<speaker>`` row.  A speaker who talks for
    the whole meeting cannot be dropped (their speech would sit unannotated
    on a headset), so such agents are synthesized on the conventional
    channel (A-D -> 0-3, the corpus rule verified on every rostered meeting)
    with a meeting-local speaker id.  Returns ``(all, synthesized)``.
    """
    present = {p.agent for p in parts}
    used_channels = {p.channel for p in parts}
    added: list[Participant] = []
    for ch, agent in enumerate(AGENTS):
        if agent in present:
            continue
        words_path = Path(words_dir) / f"{mid}.{agent}.words.xml"
        if not words_path.is_file():
            continue
        if not any(w.text and not w.punc for w in load_words(words_path)):
            continue
        if ch in used_channels:
            raise ValueError(
                f"{mid}: agent {agent} has words but channel {ch} is already "
                "taken by a rostered participant"
            )
        added.append(Participant(agent=agent, channel=ch, global_name=f"{mid}.{agent}"))
    full = sorted([*parts, *added], key=lambda p: p.channel)
    validate_participants(mid, full)
    return full, added


# --- headset audio -----------------------------------------------------------


def headset_paths(root: str | Path, meeting_id: str) -> list[Path]:
    audio_dir = Path(root) / "amicorpus" / meeting_id / "audio"
    return [audio_dir / f"{meeting_id}.Headset-{ch}.wav" for ch in range(NUM_HEADSETS)]


def _read_headset(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SOURCE_SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr}, expected {SOURCE_SAMPLE_RATE}")
    if data.shape[1] != 1:
        raise ValueError(f"{path}: {data.shape[1]} channels, expected mono")
    return data[:, 0]


def transcode_meeting(
    root: str | Path, meeting_id: str, flac_dir: str | Path
) -> Recording:
    """Write ``<flac_dir>/<MID>.flac``: four headsets as channels 0-3,
    resampled 16 -> 24 kHz (torchaudio sinc), PCM_16.  Skips the write when
    the target exists with the expected frame count.  Headsets that differ
    by more than one second are a corpus error, not something to pad over.
    """
    root, flac_dir = Path(root), Path(flac_dir)
    mono = [_read_headset(p) for p in headset_paths(root, meeting_id)]
    lengths = [m.shape[0] for m in mono]
    if max(lengths) - min(lengths) > SOURCE_SAMPLE_RATE:
        raise ValueError(
            f"{meeting_id}: headset frame count mismatch {lengths} (> 1 s apart)"
        )
    n = min(lengths)
    expected_frames = round(n * TARGET_SAMPLE_RATE / SOURCE_SAMPLE_RATE)
    out = flac_dir / f"{meeting_id}.flac"
    # torchaudio's polyphase resampler can land one frame short of the
    # rounded ratio at exact multiples (EN2002a: 51425024 vs 51425025), so
    # parity is checked to one frame, not to equality.
    if not (out.is_file() and abs(sf.info(str(out)).frames - expected_frames) <= 1):
        stacked = torch.from_numpy(np.stack([m[:n] for m in mono], axis=0))
        resampled = torchaudio.functional.resample(
            stacked, orig_freq=SOURCE_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE
        )
        if abs(resampled.shape[1] - expected_frames) > 1:
            raise RuntimeError(
                f"{meeting_id}: resampled to {resampled.shape[1]} frames, "
                f"expected {expected_frames} (+-1)"
            )
        flac_dir.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        sf.write(
            str(tmp),
            resampled.T.numpy(),
            TARGET_SAMPLE_RATE,
            subtype="PCM_16",
            format="FLAC",
        )
        tmp.replace(out)
    return load_ami_recordings(root, flac_dir, [meeting_id])[meeting_id]


def load_ami_recordings(
    root: str | Path, flac_dir: str | Path, meeting_ids: Sequence[str]
) -> dict[str, Recording]:
    root, flac_dir = Path(root), Path(flac_dir)
    out: dict[str, Recording] = {}
    for mid in meeting_ids:
        path = flac_dir / f"{mid}.flac"
        if not path.is_file():
            raise FileNotFoundError(f"{mid}: transcode first ({path} missing)")
        info = sf.info(str(path))
        if (info.channels, info.samplerate) != (NUM_HEADSETS, TARGET_SAMPLE_RATE):
            raise ValueError(
                f"{path}: {info.channels} ch @ {info.samplerate} Hz, "
                f"expected {NUM_HEADSETS} @ {TARGET_SAMPLE_RATE}"
            )
        out[mid] = Recording(
            id=mid,
            audio_relpath=str(path.relative_to(root)),
            sample_rate=TARGET_SAMPLE_RATE,
            num_channels=NUM_HEADSETS,
            duration=round(info.frames / info.samplerate, 6),
        )
    return out
