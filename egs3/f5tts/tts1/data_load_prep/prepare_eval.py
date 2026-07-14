#!/usr/bin/env python3
"""
Prepare eval sets for F5-TTS in ESPnet.

Seed-TTS (TTS task, EN + ZH):
  <out_dir>/eval_en/meta.csv       pipe-delimited: audio_file|text|prompt_audio|prompt_text
  <out_dir>/eval_zh/meta.csv

LibriSpeech test-clean:
  <out_dir>/librispeech_test_clean/meta.csv   pipe-delimited: audio_file|text
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf


# ---------------------------------------------------------------------------
# Seed-TTS
# ---------------------------------------------------------------------------
def prepare_seed_tts(db_seed_tts: Path, out_root: Path, eval_set: str):
    testset_dir = db_seed_tts / "seedtts_testset"

    meta_files = {
        "EN":      testset_dir / "en" / "meta.lst",
        "ZH":      testset_dir / "zh" / "meta.lst",
        "ZH_hard": testset_dir / "zh" / "hardcase.lst",
    }

    lang_rows: dict = {"EN": [], "ZH": []}

    for tag, meta_file in meta_files.items():
        if not meta_file.exists():
            print(f"Warning: {meta_file} not found, skipping {tag}.", file=sys.stderr)
            continue

        lang     = "EN" if tag == "EN" else "ZH"
        lang_dir = testset_dir / lang.lower()

        print(f"[Seed-TTS {tag}] Parsing {meta_file} ...")

        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) < 4:
                    continue

                filename = parts[0]
                p_text   = parts[1]
                p_audio  = parts[2]
                s_text   = parts[3]

                prompt_path = str(lang_dir / p_audio)

                # gt audio lives in wavs/<filename>.wav
                gt_wav   = lang_dir / "wavs" / f"{filename}.wav"
                wav_path = str(gt_wav) if gt_wav.exists() else filename

                lang_rows[lang].append((wav_path, s_text, prompt_path, p_text))

    for lang, rows in lang_rows.items():
        if not rows:
            continue
        out_dir  = out_root / f"{eval_set}_{lang.lower()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "meta.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("audio_file|text|prompt_audio|prompt_text\n")
            for wav, text, prompt_wav, prompt_text in rows:
                f.write(f"{wav}|{text}|{prompt_wav}|{prompt_text}\n")
        print(f"[Seed-TTS {lang}] {len(rows)} utterances → {csv_path}")


# ---------------------------------------------------------------------------
# LibriSpeech test-clean
# ---------------------------------------------------------------------------
def prepare_librispeech(db_librispeech: Path, out_root: Path, eval_set: str):
    test_clean = db_librispeech / "test-clean"
    if not test_clean.is_dir():
        print(f"Error: {test_clean} not found.", file=sys.stderr)
        sys.exit(1)

    out_dir  = out_root / f"librispeech_test_clean_{eval_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "meta.csv"

    rows = []
    skipped_missing  = 0
    skipped_duration = 0

    for trans_file in sorted(test_clean.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                utt_id = parts[0]
                text   = parts[1]
                wav    = chapter_dir / f"{utt_id}.flac"

                if not wav.exists():
                    skipped_missing += 1
                    continue

                dur = sf.info(str(wav)).duration
                if not (0.4 <= dur <= 30.0):
                    skipped_duration += 1
                    continue

                rows.append((str(wav), text))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        for wav, text in rows:
            f.write(f"{wav}|{text}\n")

    print(f"[LibriSpeech test-clean] {len(rows)} utterances → {csv_path}")
    if skipped_missing:
        print(f"  Skipped {skipped_missing} missing audio files.")
    if skipped_duration:
        print(f"  Skipped {skipped_duration} utterances outside duration bounds.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare Seed-TTS and LibriSpeech eval sets for F5-TTS"
    )
    parser.add_argument("db_seed_tts",    help="Path to seed-tts-eval root")
    parser.add_argument("db_librispeech", help="Path to LibriSpeech root")
    parser.add_argument("out_dir",        help="Output directory for eval CSVs")
    parser.add_argument("--eval_set",     default="eval")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    prepare_seed_tts(
        db_seed_tts=Path(args.db_seed_tts),
        out_root=out_root,
        eval_set=args.eval_set,
    )
    prepare_librispeech(
        db_librispeech=Path(args.db_librispeech),
        out_root=out_root,
        eval_set=args.eval_set,
    )


if __name__ == "__main__":
    main()
