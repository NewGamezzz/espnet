r"""Verify a FireRedTTS-2 run and gather its wavs for ingest.

``firered_infer.py`` already writes ``<window_id>.wav``, so unlike the
MOSS-TTSD collector this script renames nothing - the id map was never a
problem, because the runner is ours.  What is left is the job that actually
protects the results table:

* EVERY expected dialogue must be present, exactly once, with audio on disk.
  Rows are recorded rather than raised during a run, so a shard that lost
  three dialogues to their context cap exits cleanly and looks fine.
* A turn near their generation cap is FLAGGED.  Their cap is
  ``max_audio_length_ms=30_000`` on EACH turn (375 frames of 80 ms), not on
  the dialogue, so a looping turn hides inside a plausible-looking total
  duration.  This is why the runner records per-turn frame counts.
* The shards' wavs are gathered into one directory, because shards run as
  separate jobs into separate directories and the ingest's ``wav_dir`` is a
  single path.

Two attempts at the same dialogue inside one shard is a RETRY (the runner
appends, so a resumed shard keeps its history) and the last one wins.  The
same dialogue in two different shards is an ERROR: the input was split
wrong, and one of the two generations would be silently unused.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean

#: Their per-turn cap: ``max_audio_length_ms=30_000`` at 12.5 Hz.
MAX_TURN_FRAMES = 375
#: How close to the cap counts as "hit the cap".
RUNAWAY_FRACTION = 0.95


def read_shards(run_dir: Path) -> dict[str, tuple[Path, dict]]:
    """``{window_id: (shard_dir, record)}``, last attempt per shard wins."""
    paths = sorted(Path(run_dir).rglob("records.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no records.jsonl under {run_dir}")
    latest: dict[str, tuple[Path, dict]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            wid = record["window_id"]
            previous = latest.get(wid)
            if previous is not None and previous[0] != path.parent:
                raise ValueError(f"{wid}: came back from two shards")
            latest[wid] = (path.parent, record)
    return latest


def verify(
    run_dir: Path,
    expected_ids: list[str],
    out_dir: Path | None = None,
    max_turn_frames: int = MAX_TURN_FRAMES,
) -> dict:
    """Check the run, optionally gather its wavs, and report.

    Raises if any expected dialogue is missing, failed, or has no wav on
    disk.  Returns ``{"collected", "runaway", "duration_sec",
    "total_wall_sec"}``; ``runaway`` is a list of
    ``(window_id, turn_index, frames)``.
    """
    latest = read_shards(run_dir)

    missing = [wid for wid in expected_ids if wid not in latest]
    if missing:
        raise ValueError(
            f"{len(missing)} dialogues never came back: {', '.join(missing[:20])}"
        )
    failed = [wid for wid in expected_ids if latest[wid][1].get("status") != "ok"]
    if failed:
        raise ValueError(
            f"{len(failed)} dialogues failed: {', '.join(failed[:20])} "
            "(see the records for the traceback)"
        )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    ceiling = RUNAWAY_FRACTION * max_turn_frames
    runaway: list[tuple[str, int, int]] = []
    durations: list[float] = []
    wall = 0.0
    for wid in expected_ids:
        shard, record = latest[wid]
        wav = shard / f"{wid}.wav"
        if not wav.is_file():
            raise FileNotFoundError(f"{wid}: {wav}")
        if out_dir is not None:
            shutil.copyfile(wav, out_dir / wav.name)
        durations.append(float(record.get("duration_sec", 0.0)))
        wall += float(record.get("wall_sec", 0.0))
        for index, turn in enumerate(record.get("turns", [])):
            if turn["frames"] >= ceiling:
                runaway.append((wid, index, turn["frames"]))

    return {
        "collected": len(expected_ids),
        "runaway": runaway,
        "duration_sec": {
            "mean": mean(durations) if durations else 0.0,
            "min": min(durations) if durations else 0.0,
            "max": max(durations) if durations else 0.0,
        },
        "total_wall_sec": wall,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="gather the wavs here for the ingest's wav_dir",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="our manifest.jsonl; its window_ids are what must come back",
    )
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    expected = [json.loads(line)["window_id"] for line in lines if line.strip()]
    report = verify(args.run_dir, expected, out_dir=args.out_dir)
    stats = report["duration_sec"]
    print(
        f"verified {report['collected']} dialogues; duration "
        f"mean {stats['mean']:.2f} s (min {stats['min']:.2f}, "
        f"max {stats['max']:.2f}); {report['total_wall_sec'] / 3600:.2f} GPU-hours"
    )
    if args.out_dir is not None:
        print(f"gathered -> {args.out_dir}")
    if report["runaway"]:
        rows = ", ".join(f"{w}[turn {i}]={f}" for w, i, f in report["runaway"][:20])
        print(
            f"RUNAWAY TURNS (>= {RUNAWAY_FRACTION:.0%} of their "
            f"{MAX_TURN_FRAMES}-frame per-turn cap): {rows}"
        )


if __name__ == "__main__":
    main()
