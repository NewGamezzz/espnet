"""Derive the 3-speaker CoVoMix2 test set: same dialogues, same
transcriptions, plus one seeded LibriSpeech test-clean prompt per dialogue
(``audio_prompt_spk3`` + transcription) from a speaker disjoint with that
dialogue's spk1/spk2 and with duration inside the [min, max] band of the
2000 existing prompts.

The derived root keeps the original layout and index filename, so the eval
config only repoints ``testset.root`` and sets ``testset.num_channels: 3``;
round-robin turn assignment is the loader's existing ``i % num_channels``
rule.  Provenance goes to ``build_meta.json``.

Usage:
    python -m egs3.conversational.tts.local.build_covomix2_3spk \
        --testset-root downloads/covomix2-dialogue-testset \
        --librispeech-root /path/containing/test-clean \
        --out-root downloads/covomix2-dialogue-testset-3spk [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

DIALOGUE_INDEX = "dailydialog-dialogue.json"


def probe_sec(path: Path) -> float:
    """Duration without decoding (mirrors external_inference._probe_duration_sec)."""
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def prompt_speaker(rel: str) -> str:
    """Speaker id of an index prompt path ``test-clean/<spk>/<chap>/<utt>.flac``."""
    return Path(rel).parts[1]


def scan_test_clean(librispeech_root: Path) -> dict[str, list[dict]]:
    """speaker id -> [{"rel", "text", "sec"}] for every test-clean utterance.

    Walks ``*.trans.txt`` files (sorted, so the scan order - and with it the
    seeded sampling - is deterministic); a listed flac that is missing is a
    setup error, never skipped.
    """
    root = librispeech_root / "test-clean"
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root}: point --librispeech-root at the directory "
            "containing test-clean/"
        )
    utts: dict[str, list[dict]] = {}
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            flac = trans.parent / f"{utt_id}.flac"
            if not flac.is_file():
                raise FileNotFoundError(f"{flac}: listed in {trans} but missing")
            utts.setdefault(utt_id.split("-")[0], []).append(
                {
                    "rel": str(flac.relative_to(librispeech_root)),
                    "text": text,
                    "sec": probe_sec(flac),
                }
            )
    if not utts:
        raise ValueError(f"{root}: no utterances found")
    return utts


def build(
    testset_root: Path, librispeech_root: Path, out_root: Path, seed: int
) -> dict:
    testset_root = Path(testset_root)
    librispeech_root = Path(librispeech_root)
    out_root = Path(out_root)
    entries = json.loads(
        (testset_root / DIALOGUE_INDEX).read_text(encoding="utf-8")
    )
    band = [
        probe_sec(librispeech_root / e[k])
        for e in entries
        for k in ("audio_prompt_spk1", "audio_prompt_spk2")
    ]
    lo, hi = min(band), max(band)
    utts = scan_test_clean(librispeech_root)
    in_band = {
        spk: [u for u in us if lo <= u["sec"] <= hi] for spk, us in utts.items()
    }
    rng = random.Random(seed)
    out_entries = []
    for entry in entries:
        used = {
            prompt_speaker(entry["audio_prompt_spk1"]),
            prompt_speaker(entry["audio_prompt_spk2"]),
        }
        eligible = sorted(
            spk for spk, us in in_band.items() if us and spk not in used
        )
        if not eligible:
            raise ValueError(
                f"{entry['key']}: no test-clean speaker outside {sorted(used)} "
                f"has an utterance in the [{lo:.2f}, {hi:.2f}] s band"
            )
        spk = rng.choice(eligible)
        utt = rng.choice(in_band[spk])
        e = dict(entry)
        e["audio_prompt_spk3"] = utt["rel"]
        e["audio_prompt_spk3_transcription"] = utt["text"]
        out_entries.append(e)

    out_root.mkdir(parents=True, exist_ok=True)
    dst = out_root / "transcriptions"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(testset_root / "transcriptions", dst)
    (out_root / DIALOGUE_INDEX).write_text(
        json.dumps(out_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "seed": int(seed),
        "duration_band_sec": [lo, hi],
        "n_dialogues": len(out_entries),
        "source_index": str(testset_root / DIALOGUE_INDEX),
    }
    (out_root / "build_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Derive the 3-speaker CoVoMix2 test set"
    )
    ap.add_argument("--testset-root", required=True, type=Path)
    ap.add_argument("--librispeech-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    meta = build(args.testset_root, args.librispeech_root, args.out_root, args.seed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
