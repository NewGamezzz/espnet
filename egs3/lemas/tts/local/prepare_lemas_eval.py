#!/usr/bin/env python3
"""Build the LEMAS-eval manifest for the infer stage (spec section 8).

Per row of ``LEMAS-eval/eval/metadata.jsonl``: the leading words up to the
word boundary nearest ``split_frac`` of the duration (at least
``min_prompt`` seconds) become the speaker prompt, the rest becomes the
target with its exact transcript, and the language prompt is the speaker
prompt clip of a different-recording row of the same language. Clips are
written as 16 kHz FLAC under ``<out_dir>/clips``.

Manifest columns: ``utt_id, lang, target_text, spk_prompt_wav,
lang_prompt_wav, gt_target_wav``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from dataset.keys import classify_key, group_id


def _boundary(words, dur, split_frac, min_prompt):
    """Index ``k`` so the prompt is ``words[:k]``: nearest to ``split_frac*dur``."""
    target = split_frac * dur
    best = None
    for k in range(1, len(words)):
        end = words[k - 1]["end"]
        if end < min_prompt:
            continue
        if best is None or abs(end - target) < abs(words[best - 1]["end"] - target):
            best = k
    return best


def build_eval_manifest(
    metadata_jsonl, eval_audio_root, out_dir, split_frac=0.3, min_prompt=1.0, seed=0
) -> Path:
    """Write ``<out_dir>/manifest.tsv`` and the prompt/target clips.

    Args:
        metadata_jsonl: LEMAS-eval ``metadata.jsonl``.
        eval_audio_root: Directory holding the ``file_name`` audio paths.
        out_dir: Output directory.
        split_frac: Fraction of the row that the speaker prompt aims to cover.
        min_prompt: Minimum speaker-prompt length in seconds.
        seed: Seed for the language-prompt partner choice.

    Returns:
        Path of the manifest.

    Example:
        >>> build_eval_manifest(meta, root, "data/lemas_eval")
        PosixPath('data/lemas_eval/manifest.tsv')

    Note:
        Rows with a single word, or none long enough for the prompt floor,
        are skipped.
    """
    out_dir = Path(out_dir)
    clips = out_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    with Path(metadata_jsonl).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    entries = []
    for r in rows:
        words = (r.get("align") or {}).get("words", [])
        k = _boundary(words, float(r["dur"]), split_frac, min_prompt)
        if k is None:
            continue
        wav, sr = sf.read(str(Path(eval_audio_root) / r["file_name"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        p_end = int(words[k - 1]["end"] * sr)
        t_start = int(words[k]["start"] * sr)
        utt = r["key"]
        lang = utt[:2]
        spk = clips / f"{utt}_spk.flac"
        tgt = clips / f"{utt}_tgt.flac"
        sf.write(spk, wav[:p_end], sr, format="FLAC", subtype="PCM_16")
        sf.write(tgt, wav[t_start:], sr, format="FLAC", subtype="PCM_16")
        src = classify_key(utt)
        entries.append(
            dict(
                utt=utt,
                lang=lang,
                text=" ".join(w["word"] for w in words[k:]),
                spk=str(spk),
                gt=str(tgt),
                group=group_id(utt, src) or utt,
            )
        )
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for e in entries:
        by_lang[e["lang"]].append(e)
    out = out_dir / "manifest.tsv"
    with out.open("w", encoding="utf-8") as f:
        for lang, es in by_lang.items():
            for e in es:
                others = [o for o in es if o["group"] != e["group"]]
                partner = rng.choice(others) if others else e
                f.write(
                    "\t".join([e["utt"], lang, e["text"], e["spk"], partner["spk"], e["gt"]])
                    + "\n"
                )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_frac", type=float, default=0.3)
    ap.add_argument("--min_prompt", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(build_eval_manifest(a.metadata, a.audio_root, a.out_dir, a.split_frac, a.min_prompt, a.seed))
