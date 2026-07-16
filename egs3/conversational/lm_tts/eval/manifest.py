"""Eval manifest builders (Task 2): convert two kinds of dialogue ``jsonl``
files into the one manifest schema every downstream eval module consumes.

Set A (``build_manifest_sssd``) reads the mono ``dialogues.jsonl`` built by
``dataset/emit.py::emit_mono_record`` - metadata already carries
window-relative ``turns`` and per-speaker ``channel_wavs`` (Task 1), so this
builder maps that metadata directly onto the manifest schema.

Set B (``build_manifest_sft``) reads BagPiper's native SFT
``dev_multi_talker`` records. Speaker identity is not recoverable from the
caption, so ``speakers``/``ref_wavs`` stay ``None`` and each parsed turn's
``speaker``/``start``/``end`` stay ``None`` (text only, in caption order).

Manifest entry schema (shared by both sets)::

    {
      "example_id": str,
      "set": "sssd" | "sft",
      "system": str,
      "caption": str,
      "gt_wav": str,                       # absolute path
      "turns": [{"speaker": str|None, "start": float|None,
                  "end": float|None, "text": str}, ...],
      "speakers": list[str] | None,        # first-appearance order; sft: None
      "ref_wavs": dict[str, str] | None,   # speaker -> channel wav; sft: None
    }

This module never imports model/server code and performs no generation or
scoring - it is pure I/O plus dict reshaping, consumed by every downstream
generator/metric module per the eval battery's engine-agnostic design.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

_CURLY_TURN_RE = re.compile(r"“(.*?)”", re.S)
_STRAIGHT_TURN_RE = re.compile(r'"(.*?)"', re.S)


def parse_sft_turns(caption: str) -> list[str]:
    """Extract quoted-turn spans from an SFT user caption, in document order.

    Curly quotes (``“...”``) are the primary delimiter, matching the
    real SFT caption format. Straight quotes (``"..."``) are accepted as a
    fallback ONLY when no curly-quoted spans are found - the two styles are
    never mixed within one caption's result.
    """
    curly = _CURLY_TURN_RE.findall(caption)
    if curly:
        return curly
    return _STRAIGHT_TURN_RE.findall(caption)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _select(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Seeded sample down to ``limit`` (or all, if fewer exist / no limit),
    then sort by ``example_id`` for a deterministic, order-independent result.
    """
    if limit is not None and limit < len(records):
        chosen = random.Random(seed).sample(records, limit)
    else:
        chosen = list(records)
    return sorted(chosen, key=lambda r: r["example_id"])


def _find_message(messages: list[list], role: str, modality: str) -> str:
    for msg_role, msg_modality, content in messages:
        if msg_role == role and msg_modality == modality:
            return content
    raise KeyError(f"no {role!r}/{modality!r} message found in {messages!r}")


def _sssd_entry(record: dict) -> dict:
    messages = record["messages"]
    metadata = record["metadata"]
    return {
        "example_id": record["example_id"],
        "set": "sssd",
        "system": _find_message(messages, "system", "text"),
        "caption": _find_message(messages, "user", "text"),
        "gt_wav": _find_message(messages, "assistant", "audio"),
        "turns": metadata["turns"],
        "speakers": metadata["speakers"],
        "ref_wavs": metadata["channel_wavs"],
    }


def _sft_entry(record: dict, audio_root: Path) -> dict:
    example_id = record["example_id"]
    messages = record["messages"]
    caption = _find_message(messages, "user", "text")

    turn_texts = parse_sft_turns(caption)
    if not turn_texts:
        raise ValueError(
            f"example {example_id!r} has zero parsed turns in its caption "
            "(no curly- or straight-quoted spans found)"
        )

    utt_id = record["metadata"]["utt_id"]
    show = utt_id.split("_")[0]
    gt_wav = Path(audio_root) / show / f"{utt_id}.wav"
    if not gt_wav.exists():
        raise FileNotFoundError(str(gt_wav))

    turns = [
        {"speaker": None, "start": None, "end": None, "text": text}
        for text in turn_texts
    ]
    return {
        "example_id": example_id,
        "set": "sft",
        "system": _find_message(messages, "system", "text"),
        "caption": caption,
        "gt_wav": str(gt_wav),
        "turns": turns,
        "speakers": None,
        "ref_wavs": None,
    }


def build_manifest_sssd(
    dialogues_path: Path, limit: int | None = None, seed: int = 7
) -> list[dict]:
    """Set A manifest entries from a mono ``dialogues.jsonl`` (Task 1 shape)."""
    records = _read_jsonl(dialogues_path)
    selected = _select(records, limit, seed)
    return [_sssd_entry(r) for r in selected]


def build_manifest_sft(
    dialogues_path: Path,
    audio_root: Path,
    limit: int | None = None,
    seed: int = 7,
) -> list[dict]:
    """Set B manifest entries from a native SFT ``dialogues.jsonl``.

    ``gt_wav`` resolves to ``audio_root / show / f"{utt_id}.wav"`` where
    ``show = utt_id.split("_")[0]``; raises ``FileNotFoundError`` (with the
    tried path) if that file does not exist. Raises ``ValueError`` (naming
    the ``example_id``) for any record whose caption yields zero parsed
    turns - loud failure, no silent skips.
    """
    records = _read_jsonl(dialogues_path)
    selected = _select(records, limit, seed)
    return [_sft_entry(r, audio_root) for r in selected]


def write_manifest(entries: list[dict], path: Path) -> None:
    """Write ``entries`` as a JSON list, utf-8, indent 1."""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=1, ensure_ascii=False)


def load_manifest(path: Path) -> list[dict]:
    """Load a manifest previously written by ``write_manifest``."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
