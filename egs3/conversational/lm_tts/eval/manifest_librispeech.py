"""LibriSpeech test-clean manifest builder for the paper-Table-6 TTS eval.

Walks the READ-ONLY shared LibriSpeech corpus layout
(``<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac`` plus one
``<speaker>-<chapter>.trans.txt`` per chapter) and emits the same manifest
schema as ``eval.manifest`` with ``set: "librispeech"`` - single-speaker,
no timing, no reference wavs, so the battery scores WER/UTMOS only.

The system prompt is the authors' own single-speaker TTS system prompt,
verbatim from every record of their ``libritts_r_test_clean`` instruct-TTS
split (HF ``JinchuanTian/bagpipier_tts``, inspected 2026-07-18). The
caption is our paper-faithful construction of Table 6's "calm, neutral
voice, plus the text transcription" (the paper's exact wording is
unpublished). Transcripts are lowercased: LibriSpeech ships ALL-CAPS,
which is out-of-distribution for the model's tokenizer, and scoring is
case-insensitive anyway (whisper normalization).

This module never imports model/server code - pure I/O plus dict
reshaping, like ``eval.manifest``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.manifest import _select, write_manifest

SYSTEM_PROMPT = (
    "You are an advanced text-to-speech system that generates natural, "
    "expressive speech audio. When given a request describing desired "
    "voice characteristics and text to speak, first reason about how to "
    "best produce the speech, then provide a detailed description of the "
    "audio you will generate."
)

CAPTION_TEMPLATE = 'Please read the following text in a calm, neutral voice: "{text}"'


def build_manifest_librispeech(
    corpus_dir: Path, limit: int | None = None, seed: int = 0
) -> list[dict]:
    """Manifest entries for every utterance under ``corpus_dir``.

    Raises ``ValueError`` when no ``*.trans.txt`` exists (wrong dir) or a
    transcript line carries no text, and ``FileNotFoundError`` (with the
    tried path) when a transcript names a flac that does not exist - loud
    failure, no silent skips, matching ``eval.manifest`` conventions.
    """
    corpus_dir = Path(corpus_dir)
    records = []
    for trans_path in sorted(corpus_dir.glob("*/*/*.trans.txt")):
        for line in trans_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            text = text.strip()
            if not text:
                raise ValueError(
                    f"transcript line without text in {trans_path}: {utt_id!r}"
                )
            flac = trans_path.parent / f"{utt_id}.flac"
            if not flac.exists():
                raise FileNotFoundError(str(flac))
            lowered = text.lower()
            records.append(
                {
                    "example_id": f"librispeech_test_clean_{utt_id}",
                    "set": "librispeech",
                    "system": SYSTEM_PROMPT,
                    "caption": CAPTION_TEMPLATE.format(text=lowered),
                    "gt_wav": str(flac),
                    "turns": [
                        {
                            "speaker": None,
                            "start": None,
                            "end": None,
                            "text": lowered,
                        }
                    ],
                    "speakers": None,
                    "ref_wavs": None,
                }
            )
    if not records:
        raise ValueError(f"no transcript files found under {corpus_dir}")
    return _select(records, limit, seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", required=True, help="test-clean root (read-only)")
    ap.add_argument("--out", required=True, help="manifest JSON output path")
    ap.add_argument("--limit", type=int, default=None, help="pilot subset size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    entries = build_manifest_librispeech(args.corpus_dir, args.limit, args.seed)
    write_manifest(entries, args.out)
    print(f"wrote {len(entries)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
