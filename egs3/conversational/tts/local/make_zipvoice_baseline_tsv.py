r"""Emit a ZipVoice-Dialog test-list (their native tsv) from OUR built set.

The ZipVoice-Dialog baseline runs in its own repo and environment
(k2-fsa/ZipVoice), so the only thing this recipe hands it is an input file.
Building that file from OUR manifest rather than passing the archive's own
``test.tsv`` is what makes the comparison clean: the baseline then sees the
same script segmentation and the same (v2-normalized, mono) prompt audio our
model saw, and the model is the only thing left different.  ``--orig-tsv``
re-checks that against the archive row by row and prints every difference.

Its outputs come back through ``src/external_system_ingest.py``.

Row format (their "splitted prompt" format, detected per line by column
count)::

    {id}\t{spk1_prompt_text}\t{spk2_prompt_text}\t{spk1_wav}\t{spk2_wav}\t{text}

Monologue rows
--------------
The 1-channel records have one prompt, but their stereo model asserts a
STEREO wav for the 4-column merged format and our prompts are mono - so
those rows also go out in the 6-column format with prompt B = prompt A.
The text carries no ``[S2]``, so the second track should come back silent;
the ingest mode checks exactly that.  Their ids are written to
``<out>.mono_ids.txt`` so the decision is auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dialogue_text(record: dict) -> str:
    """``"[S1] hello [S2] hi"`` - the turns in conversation order."""
    return " ".join(
        "[{}] {}".format(turn["speaker"], turn["text"].strip())
        for turn in record["turns"]
    )


def _norm(text: str) -> str:
    return " ".join(text.split())


def build_rows(set_dir: Path) -> tuple[list[str], list[str]]:
    """``(tsv_rows, monologue_ids)`` for every record of the built set."""
    lines = (set_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    rows: list[str] = []
    mono_ids: list[str] = []
    for record in records:
        wid = record["window_id"]
        channels = record["channels"]
        wav_a = set_dir / channels[0]["prompt_wav"]
        text_a = channels[0]["prompt_text"].strip()
        if record["num_channels"] == 1:
            mono_ids.append(wid)
            wav_b, text_b = wav_a, text_a
        else:
            wav_b = set_dir / channels[1]["prompt_wav"]
            text_b = channels[1]["prompt_text"].strip()
        for path in (wav_a, wav_b):
            if not path.is_file():
                raise FileNotFoundError(path)
        text = dialogue_text(record)
        if not text.startswith("[S1]"):
            # Their loader asserts this too; failing here names the row.
            raise ValueError(f"{wid}: dialogue text does not start with [S1]")
        rows.append("\t".join([wid, text_a, text_b, str(wav_a), str(wav_b), text]))
    return rows, mono_ids


def diff_against_archive(rows: list[str], orig_tsv: Path) -> list[tuple[str, ...]]:
    """Row-by-row differences vs the archive tsv, as ``(id, field, a, b)``."""
    archive = {}
    for line in orig_tsv.read_text(encoding="utf-8").splitlines():
        items = line.split("\t")
        if len(items) == 6:
            archive[items[0]] = {"pa": items[1], "pb": items[2], "text": items[5]}
    diffs = []
    for row in rows:
        wid, text_a, text_b, _, _, text = row.split("\t")
        if wid not in archive:
            continue
        entry = archive[wid]
        if _norm(entry["text"]) != _norm(text):
            diffs.append((wid, "text", _norm(entry["text"]), _norm(text)))
        if _norm(entry["pa"]) != _norm(text_a) or _norm(entry["pb"]) != _norm(text_b):
            diffs.append(
                (
                    wid,
                    "prompt_text",
                    "{}|{}".format(entry["pa"], entry["pb"]),
                    "{}|{}".format(text_a, text_b),
                )
            )
    return diffs


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set-dir",
        required=True,
        type=Path,
        help="the built set, e.g. downloads/zipvoice-dialog-test-en-v2",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--orig-tsv",
        type=Path,
        default=None,
        help="the archive's own test.tsv; every difference is printed",
    )
    args = parser.parse_args()

    set_dir = args.set_dir.resolve()
    rows, mono_ids = build_rows(set_dir)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    Path(str(args.out) + ".mono_ids.txt").write_text(
        "\n".join(mono_ids) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(rows)} rows -> {args.out} "
        f"({len(mono_ids)} monologue rows, prompt duplicated)"
    )
    if args.orig_tsv is not None:
        diffs = diff_against_archive(rows, args.orig_tsv)
        print(f"differences vs {args.orig_tsv}: {len(diffs)}")
        for wid, field, archive_value, ours in diffs[:20]:
            print(f"  {wid} {field}\n    archive: {archive_value[:200]}")
            print(f"    ours   : {ours[:200]}")


if __name__ == "__main__":
    main()
