r"""No-GPU pre-flight for the FireRedTTS-2 input file.

Two of their limits can end a row, and both are knowable before a GPU-second
is spent:

* ``process_text_list`` RE-SEGMENTS any turn past 80 English words into
  several independently generated sub-turns.  We do not want that discovered
  in the output; we want it counted, because a system that silently
  re-segments our script is a fact any writeup has to state.
* ``generate`` RAISES ``ValueError`` once the context outgrows
  ``max_seq_len - max_generation_len`` = 3100 - 375 = 2725 frames.  Context
  is the prompts plus every turn generated so far, so a long dialogue walks
  towards that ceiling as it goes.

Run it in THEIR environment, from their repo, so the splitter checked is the
one that will run::

    cd /work/hdd/bbjs/ttrachu/development/FireRedTTS2
    .pixi/envs/default/bin/python <recipe>/local/firered_preflight.py \
        --input input_v2.jsonl --set-dir <built set>

The context number is an ESTIMATE and says so: audio frames come from the
reference duration (their generation will differ), and text frames from a
characters-per-token heuristic rather than their tokenizer.  It is a
tripwire for the tail of the set, not a measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Their tokenizer rate.
FRAMES_PER_SEC = 12.5
#: ``max_seq_len`` (3100) minus ``max_generation_len`` (30_000 ms / 80 ms).
CONTEXT_CEILING = 3100 - 375
#: Rough english characters per Qwen token; used only for the estimate.
CHARS_PER_TOKEN = 3.5
#: ``speaker + <|text_start|> + text + <|text_end|>`` costs a few tokens
#: beyond the text itself, per turn.
TOKENS_PER_TURN_OVERHEAD = 4
#: Their tag vocabulary.
SPEAKER_TAGS = ("[S1]", "[S2]", "[S3]", "[S4]")


def estimate_context_frames(
    prompt_secs: list[float], target_sec: float, texts: list[str]
) -> dict:
    """Frames of context a whole dialogue would occupy, roughly.

    Prompt audio, generated audio and text tokens are all rows of the same
    sequence, so all three count.
    """
    prompt_frames = int(sum(prompt_secs) * FRAMES_PER_SEC)
    target_frames = int(target_sec * FRAMES_PER_SEC)
    text_frames = sum(
        int(len(text) / CHARS_PER_TOKEN) + TOKENS_PER_TURN_OVERHEAD for text in texts
    )
    return {
        "prompt_frames": prompt_frames,
        "target_frames": target_frames,
        "text_frames": text_frames,
        "frames": prompt_frames + target_frames + text_frames,
    }


def check_rows(
    rows: list[dict],
    splitter,
    duration_of,
    target_secs: dict[str, float] | None = None,
) -> dict:
    """Run ``splitter`` over every row and estimate its context.

    ``splitter`` is their ``process_text_list``; ``duration_of`` maps a wav
    path to seconds; ``target_secs`` maps a dialogue id to its reference
    duration (absent = 0, i.e. text and prompts only).
    """
    target_secs = target_secs or {}
    reports: list[dict] = []
    split_rows: list[str] = []
    over: list[str] = []
    for row in rows:
        wid = row["window_id"]
        after = list(splitter(row["text_list"]))
        for text in after:
            if text[:4] not in SPEAKER_TAGS:
                raise ValueError(
                    f"{wid}: their splitter dropped the tag: {text[:20]!r}"
                )
        estimate = estimate_context_frames(
            [duration_of(path) for path in row["prompt_wav_list"]],
            target_secs.get(wid, 0.0),
            after + row["prompt_text_list"],
        )
        if len(after) != len(row["text_list"]):
            split_rows.append(wid)
        if estimate["frames"] > CONTEXT_CEILING:
            over.append(wid)
        reports.append(
            {
                "window_id": wid,
                "num_turns_in": len(row["text_list"]),
                "num_turns_after": len(after),
                "est_context_frames": estimate["frames"],
                **estimate,
            }
        )
    return {"rows": reports, "split_rows": split_rows, "over_ceiling": over}


def target_durations(set_dir: Path) -> dict[str, float]:
    """Return the reference duration per id, from the built set's GT wavs."""
    import soundfile

    lines = (set_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    out: dict[str, float] = {}
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        best = 0.0
        for channel in record["channels"]:
            rel = channel.get("gt_wav")
            if rel is None:
                continue
            info = soundfile.info(str(set_dir / rel))
            best = max(best, info.frames / info.samplerate)
        out[record["window_id"]] = best
    return out


def main() -> None:
    """CLI entry point - run from their repo, in their environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--set-dir",
        type=Path,
        default=None,
        help="the built set, for reference durations (recommended)",
    )
    args = parser.parse_args()

    import soundfile
    from fireredtts2.utils.spliter import process_text_list

    lines = args.input.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    def duration_of(path):
        info = soundfile.info(str(path))
        return info.frames / info.samplerate

    targets = target_durations(args.set_dir.resolve()) if args.set_dir else None
    report = check_rows(rows, process_text_list, duration_of, target_secs=targets)

    frames = [row["est_context_frames"] for row in report["rows"]]
    print(f"{len(rows)} rows checked, every speaker tag survived their splitter")
    print(
        f"turns re-segmented by their 80-word rule: {len(report['split_rows'])} rows"
        + (f" ({', '.join(report['split_rows'][:20])})" if report["split_rows"] else "")
    )
    print(
        f"estimated context frames: max {max(frames)}, "
        f"mean {sum(frames) // len(frames)}, ceiling {CONTEXT_CEILING}"
    )
    if report["over_ceiling"]:
        print(
            f"OVER THE CEILING, generate() would raise: "
            f"{', '.join(report['over_ceiling'])}"
        )
    else:
        print("no row is estimated over the ceiling")


if __name__ == "__main__":
    main()
