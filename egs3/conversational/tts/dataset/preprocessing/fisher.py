"""Fisher corpus ingestion: sidon-manifest parsing, text cleaning, A/B merge.

Fisher English ships 11,699 two-party telephone calls with lhotse manifests.
We consume cornell2's Sidon-restored 24 kHz audio (one mono FLAC per channel,
``fisher_wavs_sidon_24k/<shard>/<id>-{A,B}.flac``) and the ``fixed/``
supervisions (durations clamped into recording bounds).  The supervisions are
field-compatible with the SSSD parser, so ``sssd.load_supervisions`` is
reused verbatim by the builder; this module holds only what is
Fisher-specific: the two-source recordings loader (which points windows at
merged stereo FLACs; the manifests' absolute ``/scratch/...`` prefixes are
machine-specific and untrusted), the one-time A/B -> stereo ffmpeg merge
(the pipeline assumes one multi-channel file per session), and the LDC
transcript cleaning (event tags, unclear markers, acronym underscores).
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

from .sssd import Recording, Supervision, _iter_jsonl_gz

# Documented source layout for the restored audio, relative to dataset_root.
SIDON_AUDIO_SUBDIR = "fisher_wavs_sidon_24k"

_TAG_RE = re.compile(r"\[[^\]]*\]")
# Characters the frozen English charset cannot carry; an utterance containing
# any of them holds real speech we cannot transcribe faithfully, so the whole
# utterance is dropped and its span blocks windows.
_UNREPRESENTABLE = set("0123456789&*<>")


@dataclass(frozen=True)
class CleanResult:
    """``text`` is the cleaned transcript (possibly empty).

    ``unintelligible`` marks utterances whose SPEECH content is lost to the
    transcript (empty ``(( ))`` unclear markers, foreign-language spans,
    unrepresentable characters): their time spans must not survive into
    training windows.  A tag-only utterance ("[laughter]") also cleans to
    empty but is NOT unintelligible: it never carried words, so dropping it
    merely leaves non-speech audio untranscribed (accepted cost, same as
    SSSD pseudo-label gaps).
    """

    text: str
    unintelligible: bool


def clean_fisher_text(text: str) -> CleanResult:
    """Normalize one LDC Fisher transcript line (see ``CleanResult``)."""
    if any(c in _UNREPRESENTABLE for c in text):
        return CleanResult("", True)
    had_unclear = "((" in text
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = cleaned.replace("(", " ").replace(")", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())
    if not cleaned and had_unclear:
        return CleanResult("", True)
    return CleanResult(cleaned, False)


def clean_fisher_supervisions(
    sups: Sequence[Supervision],
) -> tuple[list[Supervision], list[tuple[float, float]], int]:
    """Apply ``clean_fisher_text`` per utterance, BEFORE turn merging.

    Returns ``(kept, unintelligible_spans, n_benign_dropped)``.  Cleaning
    must precede ``merge_turns``: dropped utterances would otherwise be
    merged across invisibly, hiding unintelligible speech inside a turn
    whose text does not cover it.
    """
    kept: list[Supervision] = []
    spans: list[tuple[float, float]] = []
    n_benign = 0
    for s in sups:
        res = clean_fisher_text(s.text)
        if res.unintelligible:
            spans.append((s.start, s.end))
        elif not res.text:
            n_benign += 1
        else:
            kept.append(dataclasses.replace(s, text=res.text))
    return kept, spans, n_benign
