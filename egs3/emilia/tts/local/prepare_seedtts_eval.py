#!/usr/bin/env python3
"""Build Seed-TTS eval manifests: test-en, test-zh and test-hard.

The staged prepare_eval.py folded hardcase.lst into the zh bucket, producing
one 2,420-row file. The paper reports test-zh (2,020) and test-hard (400)
separately, and test-hard is the interesting one.

Seed-TTS meta.lst columns are:
    filename | prompt_text | prompt_audio | target_text
"""

from __future__ import annotations

import argparse
from pathlib import Path

_SETS = (
    ("test_en", "en", "meta.lst"),
    ("test_zh", "zh", "meta.lst"),
    ("test_hard", "zh", "hardcase.lst"),
)


def _parse(lst_path: Path, lang_dir: Path) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    with lst_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 4:
                continue
            utt_id, prompt_text, prompt_audio, target_text = parts[:4]
            wav = lang_dir / "wavs" / f"{utt_id}.wav"
            rows.append((
                utt_id,
                str(wav.resolve()),
                target_text,
                str((lang_dir / prompt_audio).resolve()),
                prompt_text,
            ))
    return rows


def prepare_seedtts(testset_dir: str | Path, out_dir: str | Path) -> dict:
    """Write the three eval manifests. Returns {set_name: row_count}."""
    testset_dir = Path(testset_dir)
    out_dir = Path(out_dir)
    counts: dict[str, int] = {}
    for name, lang, lst in _SETS:
        lang_dir = testset_dir / lang
        lst_path = lang_dir / lst
        if not lst_path.is_file():
            raise FileNotFoundError(f"Seed-TTS list not found: {lst_path}")
        rows = _parse(lst_path, lang_dir)
        target = out_dir / name
        target.mkdir(parents=True, exist_ok=True)
        with (target / "meta.tsv").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write("\t".join(row) + "\n")
        counts[name] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testset_dir", help="path to seedtts_testset")
    parser.add_argument("out_dir", help="output root for eval manifests")
    args = parser.parse_args()
    for name, n in prepare_seedtts(args.testset_dir, args.out_dir).items():
        print(f"{name}: {n} utterances")


if __name__ == "__main__":
    main()
