"""Derive the 3-speaker CoVoMix2 test set: same dialogues, same
transcriptions, plus one seeded LibriSpeech test-clean prompt per dialogue
(``audio_prompt_spk3`` + transcription, lowercased to match the native
spk1/spk2 transcription convention) from a speaker disjoint with that
dialogue's spk1/spk2. The candidate duration band is per-dialogue -
``[0.75 * min(d1, d2), 1.25 * max(d1, d2)]`` for that dialogue's own spk1/
spk2 durations - falling back to the global ``[p10, p90]`` band of the 2000
native prompt durations when no eligible speaker has an utterance in the
per-dialogue band.

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

    # One probe pass, keeping each dialogue's own (d1, d2) pair as well as
    # the flattened list used for the global fallback band.
    per_dialogue_durs = []
    global_durs = []
    for entry in entries:
        d1 = probe_sec(librispeech_root / entry["audio_prompt_spk1"])
        d2 = probe_sec(librispeech_root / entry["audio_prompt_spk2"])
        per_dialogue_durs.append((d1, d2))
        global_durs.extend([d1, d2])

    sorted_durs = sorted(global_durs)
    n = len(sorted_durs)
    fallback_lo = sorted_durs[int(0.10 * (n - 1))]
    fallback_hi = sorted_durs[int(0.90 * (n - 1))]

    utts = scan_test_clean(librispeech_root)

    def eligible_speakers(lo: float, hi: float, used: set[str]) -> list[str]:
        return sorted(
            spk
            for spk, us in utts.items()
            if spk not in used and any(lo <= u["sec"] <= hi for u in us)
        )

    rng = random.Random(seed)
    out_entries = []
    n_fallback = 0
    for entry, (d1, d2) in zip(entries, per_dialogue_durs):
        used = {
            prompt_speaker(entry["audio_prompt_spk1"]),
            prompt_speaker(entry["audio_prompt_spk2"]),
        }
        lo, hi = 0.75 * min(d1, d2), 1.25 * max(d1, d2)
        eligible = eligible_speakers(lo, hi, used)
        fallback = False
        if not eligible:
            fallback = True
            lo, hi = fallback_lo, fallback_hi
            eligible = eligible_speakers(lo, hi, used)
        if not eligible:
            raise ValueError(
                f"{entry['key']}: no test-clean speaker outside {sorted(used)} "
                f"has an utterance in the [{lo:.2f}, {hi:.2f}] s band"
            )
        spk = rng.choice(eligible)
        candidates = [u for u in utts[spk] if lo <= u["sec"] <= hi]
        utt = rng.choice(candidates)
        if fallback:
            n_fallback += 1
        e = dict(entry)
        e["audio_prompt_spk3"] = utt["rel"]
        e["audio_prompt_spk3_transcription"] = utt["text"].lower()
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
        "band_rule": (
            "per-dialogue [0.75*min, 1.25*max], fallback global [p10, p90]"
        ),
        "global_fallback_band_sec": [fallback_lo, fallback_hi],
        "n_fallback": n_fallback,
        "transcription_case": "lower",
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
