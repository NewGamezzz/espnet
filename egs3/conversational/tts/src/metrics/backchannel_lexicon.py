"""Lexical backchannel rule for the Talking Turns judge labels.

The paper (Arora et al., ICLR 2025, Sec. 4.2 and A.4) does not annotate
backchannels on collected dialogues by duration: every listener utterance
is transcribed with Whisper and, following Wang et al. (2024), an utterance
whose text is one of the most frequent *isolated one- or two-word phrases*
of the Switchboard backchannel annotations (Ekstedt's ``backchannels.csv``)
is a backchannel when it happens during the other speaker's turn. The judge
checkpoint was trained on labels made the same way, so this is the
definition the judge learned.

This module provides that rule as a labeler that plugs into
``TurnTakingJudgeMetric`` (``bc_rule: lexical``):

* :func:`normalize_phrase` - the ONE normaliser applied to the lexicon and
  to the ASR output. It is deliberately not whisper's
  ``EnglishTextNormalizer``: that one deletes the hesitation class
  (hmm / mm / mhm / uh / um) for WER purposes, which is exactly the class a
  backchannel lexicon has to keep.
* :func:`build_lexicon` / :func:`load_lexicon` - derive the phrase list
  from ``backchannels.csv`` (``words`` column, which is already stripped of
  ``[noise]``/``[laughter]`` tokens), or read the committed copy
  ``local/backchannel_lexicon.txt``.
* :class:`LexicalBackchannelLabeler` - transcribes each candidate listener
  IPU (short enough to be a 1-2 word phrase) and relabels it ``BC`` when
  the normalised transcript EQUALS a lexicon phrase and the IPU starts
  while the other channel holds the floor. No duration cap and no
  "floor does not pass" condition: the paper has neither.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Span = Tuple[float, float]

DEFAULT_MIN_COUNT = 100  # 36 phrases, 94.1 % of the 60,917 annotated backchannels
DEFAULT_MAX_WORDS = 2  # "isolated one and two-word phrases" (paper A.4)
DEFAULT_MAX_IPU_SEC = 3.0  # transcription-candidate filter only (compute), not a rule

# Whisper spellings -> Switchboard spellings, per token.
_TOKEN_MAP = {
    "mm-hmm": "um-hum",
    "mmhmm": "um-hum",
    "mm-hm": "um-hum",
    "mhm": "um-hum",
    "mhmm": "um-hum",
    "umhum": "um-hum",
    "um-hmm": "um-hum",
    "uh-hum": "um-hum",
    "uhhuh": "uh-huh",
    "uh-hah": "uh-huh",
    "hmm": "hm",
    "hmmm": "hm",
    "mm": "hm",
    "mmm": "hm",
    "ok": "okay",
    "yea": "yeah",
    "ya": "yeah",
    "yup": "yep",
}
# Two-token Whisper renderings of one Switchboard token.
_PHRASE_MAP = {
    "uh huh": "uh-huh",
    "mm hmm": "um-hum",
    "um hum": "um-hum",
    "mm hm": "um-hum",
    "uh hum": "um-hum",
}

_BRACKETS = re.compile(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)")
_PUNCT = re.compile(r"[^a-z' \-]")


def normalize_phrase(text: str) -> str:
    """Lower-case, drop bracketed annotations and punctuation (internal
    apostrophes and hyphens kept), map Whisper spellings onto Switchboard
    ones, collapse whitespace."""
    t = _BRACKETS.sub(" ", text.lower())
    t = _PUNCT.sub(" ", t)
    toks = [w.strip("-'") for w in t.split()]
    toks = [_TOKEN_MAP.get(w, w) for w in toks if w]
    phrase = " ".join(toks)
    for src, dst in _PHRASE_MAP.items():
        phrase = re.sub(rf"\b{re.escape(src)}\b", dst, phrase)
    return " ".join(phrase.split())


def build_lexicon(
    csv_path: Path,
    min_count: int = DEFAULT_MIN_COUNT,
    max_words: int = DEFAULT_MAX_WORDS,
) -> List[Tuple[str, int]]:
    """Phrase list from Ekstedt's ``backchannels.csv``: normalised ``words``
    column, at most ``max_words`` words, kept when annotated at least
    ``min_count`` times. Returns ``[(phrase, count)]`` by falling count."""
    counts: Dict[str, int] = {}
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            words = ast.literal_eval(row["words"])
            phrase = normalize_phrase(" ".join(words))
            if not phrase or len(phrase.split()) > max_words:
                continue
            counts[phrase] = counts.get(phrase, 0) + 1
    kept = [(p, c) for p, c in counts.items() if c >= min_count]
    return sorted(kept, key=lambda pc: (-pc[1], pc[0]))


def write_lexicon(entries: Iterable[Tuple[str, int]], path: Path, header: str) -> None:
    lines = [f"# {h}" for h in header.splitlines()]
    lines += [f"{p}\t{c}" for p, c in entries]
    Path(path).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")


def load_lexicon(path: Path) -> frozenset:
    """Read a lexicon file (``phrase<TAB>count`` lines, ``#`` comments);
    phrases are re-normalised so the file and the ASR side always agree."""
    out = set()
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        phrase = normalize_phrase(ln.split("\t")[0])
        if phrase:
            out.add(phrase)
    if not out:
        raise ValueError(f"empty backchannel lexicon: {path}")
    return frozenset(out)


def default_lexicon_path() -> Path:
    """``local/backchannel_lexicon.txt`` of this recipe."""
    return Path(__file__).resolve().parents[2] / "local" / "backchannel_lexicon.txt"


def other_holds_floor(t: float, mine: Sequence[Span], other: Sequence[Span]) -> bool:
    """True if the OTHER channel holds the floor at time ``t`` (same rule as
    the duration proxy in :mod:`.turn_taking_judge`)."""
    if any(s <= t < e for s, e in other):
        return True
    other_last = max((e for _, e in other if e <= t), default=None)
    mine_last = max((e for _, e in mine if e <= t), default=None)
    return other_last is not None and (mine_last is None or other_last > mine_last)


class LexicalBackchannelLabeler:
    """Relabel listener IPUs as ``BC`` by transcript, the paper's rule.

    ``transcriber(wav, sr) -> str`` is any callable with the ASR metric's
    transcriber signature (default there: faster-whisper large-v3); pass it
    with ``vad_filter=False`` so short, quiet backchannels are not dropped
    before they are heard.

    ``max_ipu_sec`` only bounds which IPUs are transcribed at all (a 1-2
    word phrase does not last 3 s); it is a compute filter, not part of the
    rule, and is recorded per span in the returned records.
    """

    def __init__(
        self,
        transcriber: Callable[[np.ndarray, int], str],
        lexicon: Optional[Iterable[str]] = None,
        lexicon_path: Optional[Path] = None,
        max_ipu_sec: float = DEFAULT_MAX_IPU_SEC,
    ) -> None:
        if lexicon is not None:
            self.lexicon = frozenset(normalize_phrase(p) for p in lexicon)
        else:
            self.lexicon = load_lexicon(lexicon_path or default_lexicon_path())
        self.transcriber = transcriber
        self.max_ipu_sec = max_ipu_sec

    def transcribe_spans(
        self,
        channel_spans: Sequence[Sequence[Span]],
        channel_wavs: Sequence[np.ndarray],
        sr: int,
    ) -> List[List[Dict[str, Any]]]:
        """Per channel, one record per IPU: ``{start, end, text, candidate}``
        (``text`` is the raw transcript, ``None`` when not a candidate).
        This is the part worth caching; :meth:`label_records` is free."""
        chans = [sorted((float(s), float(e)) for s, e in ch) for ch in channel_spans]
        out: List[List[Dict[str, Any]]] = []
        for wav, mine in zip(channel_wavs, chans):
            recs = []
            for s, e in mine:
                cand = e - s <= self.max_ipu_sec + 1e-9
                text = None
                if cand:
                    a, b = int(round(s * sr)), int(round(e * sr))
                    text = self.transcriber(np.asarray(wav[a:b], dtype=np.float32), sr)
                recs.append({"start": s, "end": e, "candidate": cand, "text": text})
            out.append(recs)
        return out

    def label_records(
        self, records: Sequence[Sequence[Dict[str, Any]]]
    ) -> List[List[Tuple[Span, str]]]:
        """Apply the rule to transcript records. Returns the
        ``apply_backchannel_proxy`` shape: per channel ``[((s, e), kind)]``
        with kind ``IPU`` or ``BC``. Two channels only; other counts are
        returned as all-``IPU``. Records are annotated in place with
        ``normalized`` and ``is_bc``."""
        chans = [[(r["start"], r["end"]) for r in ch] for ch in records]
        out: List[List[Tuple[Span, str]]] = []
        for idx, recs in enumerate(records):
            labelled: List[Tuple[Span, str]] = []
            for r in recs:
                kind = "IPU"
                norm = normalize_phrase(r["text"]) if r.get("text") else ""
                r["normalized"] = norm
                if (
                    len(chans) == 2
                    and norm
                    and norm in self.lexicon
                    and other_holds_floor(r["start"], chans[idx], chans[1 - idx])
                ):
                    kind = "BC"
                r["is_bc"] = kind == "BC"
                labelled.append(((r["start"], r["end"]), kind))
            out.append(labelled)
        return out
