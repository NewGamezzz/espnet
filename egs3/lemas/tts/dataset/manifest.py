"""Manifest schema for the LEMAS recipe (spec section 3.5).

Column 3 is the target phone string so espnet3's stock ``create_token_list``
(which reads ``parts[2]``) needs no change. ``ManifestColumns`` keeps 30 M
rows as numpy columns and offset buffers, as the Emilia recipe does.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from dataset.keys import SOURCES, segment_index
from src.text.lemas_phonemizer import LANGS

COLUMNS = [
    "utt_id",
    "audio",
    "phones",
    "lang",
    "source",
    "group",
    "dur",
    "jsonl_path",
    "byte_offset",
    "spk_mode",
    "word_bounds",
    "phones_by_word",
]
SPK_MODES = ["none", "group", "split"]


@dataclass
class ManifestRow:
    """One manifest line; field order is the column order."""

    utt_id: str
    audio: str
    phones: str
    lang: str
    source: str
    group: str
    dur: float
    jsonl_path: str
    byte_offset: int
    spk_mode: str
    word_bounds: str
    phones_by_word: str

    def to_line(self) -> str:
        """Serialize as one tab-separated line (no newline)."""
        return "\t".join(str(getattr(self, f.name)) for f in fields(self))


def write_manifest(rows: Iterable[ManifestRow], path) -> int:
    """Write rows to ``path``; returns the row count.

    Args:
        rows: Manifest rows in output order.
        path: Destination tsv (parent directories are created).

    Returns:
        Number of rows written.

    Example:
        >>> write_manifest([row], "data/manifest/train.tsv")
        1
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.to_line() + "\n")
            n += 1
    return n


class _StrColumn:
    """Variable-length strings in one bytes buffer plus int64 offsets."""

    def __init__(self, chunks: List[bytes]):
        self.buf = b"".join(chunks)
        lens = np.fromiter((len(c) for c in chunks), dtype=np.int64, count=len(chunks))
        self.off = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lens)])

    def __call__(self, i: int) -> str:
        return self.buf[self.off[i] : self.off[i + 1]].decode("utf-8")


class ManifestColumns:
    """Column-oriented, memory-compact view of a manifest.

    Attributes (after :meth:`load`): ``n_rows``, ``utt_id`` (S64 array),
    ``lang`` / ``source`` / ``spk_mode`` (int8 indexes into ``LANGS`` /
    ``SOURCES`` / ``SPK_MODES``), ``group`` (int32 index into
    ``group_names``, -1 for none), ``dur`` (float32), ``seg`` (int32, -1 when
    the source has no segment index); accessors ``audio(i)``, ``phones(i)``,
    ``word_bounds(i)``, ``phones_by_word(i)``.
    """

    def __init__(self, **cols):
        self.__dict__.update(cols)

    @classmethod
    def load(cls, path) -> "ManifestColumns":
        """Load a manifest written by :func:`write_manifest`.

        Args:
            path: Manifest tsv.

        Returns:
            A :class:`ManifestColumns`.

        Raises:
            ValueError: On a line with the wrong column count.
            RuntimeError: If the manifest is empty.

        Example:
            >>> cols = ManifestColumns.load("data/manifest/valid.tsv")
            >>> cols.audio(0)
            'de/de000/....flac'
        """
        utt, audio, phones, lang, source, group, dur, seg, mode, wb, pbw = (
            [] for _ in range(11)
        )
        group_index: dict = {}
        with Path(path).open("rb") as f:
            for line in f:
                parts = line.rstrip(b"\n").split(b"\t")
                if len(parts) != len(COLUMNS):
                    raise ValueError(f"bad manifest line: {line[:80]!r}")
                key = parts[0].decode()
                src = parts[4].decode()
                utt.append(parts[0])
                audio.append(parts[1])
                phones.append(parts[2])
                lang.append(LANGS.index(parts[3].decode()))
                source.append(SOURCES.index(src))
                g = parts[5].decode()
                group.append(group_index.setdefault(g, len(group_index)) if g else -1)
                dur.append(float(parts[6]))
                s = segment_index(key, src)
                seg.append(-1 if s is None else s)
                mode.append(SPK_MODES.index(parts[9].decode()))
                wb.append(parts[10])
                pbw.append(parts[11])
        if not utt:
            raise RuntimeError(f"Manifest is empty: {path}")
        return cls(
            n_rows=len(utt),
            utt_id=np.array(utt, dtype="S64"),
            audio=_StrColumn(audio),
            phones=_StrColumn(phones),
            lang=np.array(lang, dtype=np.int8),
            source=np.array(source, dtype=np.int8),
            group=np.array(group, dtype=np.int32),
            dur=np.array(dur, dtype=np.float32),
            seg=np.array(seg, dtype=np.int32),
            spk_mode=np.array(mode, dtype=np.int8),
            _wb=_StrColumn(wb),
            _pbw=_StrColumn(pbw),
            group_names=list(group_index),
        )

    def word_bounds(self, i: int) -> List[Tuple[float, float]]:
        """Per-word ``(start, end)`` seconds for split rows, else ``[]``."""
        s = self._wb(i)
        if not s:
            return []
        return [tuple(float(v) for v in x.split(":")) for x in s.split(",")]

    def phones_by_word(self, i: int) -> List[List[str]]:
        """Per-word phone lists for split rows, else ``[]``."""
        s = self._pbw(i)
        return [w.split(" ") for w in s.split("|")] if s else []
