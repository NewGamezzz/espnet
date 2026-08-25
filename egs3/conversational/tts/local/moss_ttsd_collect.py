r"""Collect a MOSS-TTSD run into ``<dialogue_id>.wav``, ready for ingest.

MOSS-TTSD names its output by INPUT LINE NUMBER
(``generation_utils.py::prepare_sample``: ``sample_id = f"{line_no:06d}"``),
so nothing in the wav directory identifies the dialogue.  What does identify
it is ``output.jsonl``: ``_make_output_record`` copies every unrecognised
input field through, so the ``window_id`` our converter wrote comes back
beside the absolute ``output_audio`` path and the generated ``duration``.

Reading the id out of the record rather than trusting line order means a
re-run shard, a reordered input or a partial retry cannot silently produce a
scrambled results table.

Two checks are the point of this script:

* EVERY expected dialogue must come back.  Their inference loop catches a
  failing sample, logs a warning and moves on, and writes
  ``output_audio: null`` for a sample that decodes to nothing - so a short
  table is their default failure mode, not an unusual one.
* A generation near the token cap is FLAGGED, never silently kept as a
  normal row.  ``max_new_tokens`` defaults to 8192, about 655 s at their
  stated 12.5 tokens/s, against a set whose dialogues average about 24 s.
  An autoregressive model looping is a result about the model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: Their README's rate, used only to turn the token cap into seconds for
#: the runaway check.
TOKENS_PER_SEC = 12.5
#: How close to the cap counts as "hit the cap".
RUNAWAY_FRACTION = 0.95


def read_records(run_dir: Path) -> list[dict]:
    """Every record of every ``output.jsonl`` under ``run_dir``."""
    paths = sorted(run_dir.rglob("output.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no output.jsonl under {run_dir}")
    records: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def collect(
    run_dir: Path,
    out_dir: Path,
    expected_ids: list[str],
    max_new_tokens: int = 8192,
    tokens_per_sec: float = TOKENS_PER_SEC,
) -> dict:
    """Rename their output into ``out_dir/<window_id>.wav``.

    Raises if any expected dialogue is missing, produced no audio, or came
    back twice.  Returns ``{"collected", "runaway", "cap_sec"}``.
    """
    cap_sec = max_new_tokens / tokens_per_sec
    out_dir.mkdir(parents=True, exist_ok=True)

    by_id: dict[str, dict] = {}
    for record in read_records(run_dir):
        wid = record.get("window_id")
        if wid is None:
            raise ValueError(f"record without window_id: {record.get('id')}")
        if wid in by_id:
            raise ValueError(f"{wid}: came back more than once")
        by_id[wid] = record

    missing = [wid for wid in expected_ids if wid not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} dialogues never came back: {', '.join(missing[:20])}"
        )

    runaway: list[str] = []
    for wid in expected_ids:
        record = by_id[wid]
        audio = record.get("output_audio")
        if not audio:
            raise ValueError(f"{wid}: generated no audio (output_audio is null)")
        source = Path(audio)
        if not source.is_file():
            raise FileNotFoundError(f"{wid}: {source}")
        shutil.copyfile(source, out_dir / f"{wid}.wav")
        if float(record.get("duration", 0.0)) >= RUNAWAY_FRACTION * cap_sec:
            runaway.append(wid)

    return {"collected": len(expected_ids), "runaway": runaway, "cap_sec": cap_sec}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="our manifest.jsonl; its window_ids are what must come back",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    expected = [json.loads(line)["window_id"] for line in lines if line.strip()]
    report = collect(
        args.run_dir, args.out_dir, expected, max_new_tokens=args.max_new_tokens
    )
    print(f"collected {report['collected']} -> {args.out_dir}")
    if report["runaway"]:
        print(
            f"RUNAWAY (>= {RUNAWAY_FRACTION:.0%} of the "
            f"{report['cap_sec']:.0f} s token cap): "
            f"{', '.join(report['runaway'])}"
        )


if __name__ == "__main__":
    main()
