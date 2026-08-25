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


def _parse(lst_path: Path, lang_dir: Path) -> tuple[list[tuple[str, ...]], int]:
    """Parse one Seed-TTS ``.lst`` file.

    Returns ``(rows, n_skipped)``. Splits with ``line.split("|", 3)``
    (maxsplit=3), not a bare ``split("|")`` sliced to ``parts[:4]``: a
    literal ``|`` inside ``target_text`` (the last column) would otherwise
    silently truncate the row instead of being retained as part of the
    text. Malformed lines (fewer than 4 fields) are counted, not silently
    dropped -- the row counts are this recipe's acceptance criterion
    (README), so a silent skip could make a wrong result look right.
    """
    rows: list[tuple[str, ...]] = []
    n_skipped = 0
    with lst_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = [p.strip() for p in stripped.split("|", 3)]
            if len(parts) < 4:
                n_skipped += 1
                continue
            utt_id, prompt_text, prompt_audio, target_text = parts
            wav = lang_dir / "wavs" / f"{utt_id}.wav"
            rows.append(
                (
                    utt_id,
                    str(wav.resolve()),
                    target_text,
                    str((lang_dir / prompt_audio).resolve()),
                    prompt_text,
                )
            )
    return rows, n_skipped


def prepare_seedtts(testset_dir: str | Path, out_dir: str | Path) -> dict:
    """Write the three eval manifests. Returns {set_name: row_count}.

    Raises RuntimeError if any ``.lst`` line was malformed and skipped:
    the exact row counts (test_en 1088, test_zh 2020, test_hard 400) are
    this recipe's documented acceptance criterion, so silently proceeding
    on a short count could make a wrong result look right.
    """
    testset_dir = Path(testset_dir)
    out_dir = Path(out_dir)
    counts: dict[str, int] = {}
    for name, lang, lst in _SETS:
        lang_dir = testset_dir / lang
        lst_path = lang_dir / lst
        if not lst_path.is_file():
            raise FileNotFoundError(f"Seed-TTS list not found: {lst_path}")
        rows, n_skipped = _parse(lst_path, lang_dir)
        if n_skipped:
            raise RuntimeError(
                f"{lst_path}: {n_skipped} malformed line(s) skipped "
                "(fewer than 4 '|'-separated fields). Row counts are this "
                "recipe's acceptance criterion; fix the source file rather "
                "than silently proceeding on a short count."
            )
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
