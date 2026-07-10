"""Dump a handful of training windows for human inspection.

For each sampled window this writes:
  - ``<window_id>_mix.wav``: mono mixdown (mean over channels) at the training
    sample rate, for quick listening;
  - ``<window_id>.txt``: per-branch masked text rendered with ``<turn>`` as
    ``|`` and ``<OTHER>`` as ``#``, plus the raw turn table, so the masking is
    eyeballable next to the audio.

Usage (from the recipe dir, after the builder ran):
    python local/dump_debug.py --split valid --num-windows 5 --out-dir exp/debug_dump
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soundfile as sf  # noqa: E402

from egs3.conversational.tts.dataset.dataset import ConversationDataset  # noqa: E402
from egs3.conversational.tts.dataset.preprocessing.text import (  # noqa: E402
    build_branch_texts,
    render_tokens,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("exp/debug_dump"))
    parser.add_argument(
        "--recipe-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--window-id",
        action="append",
        default=None,
        help="dump these window ids instead of sampling (repeatable)",
    )
    args = parser.parse_args()

    ds = ConversationDataset(
        split=args.split,
        recipe_dir=args.recipe_dir,
        dataset_root=args.dataset_root,
        permute_channels=False,
    )
    if args.window_id:
        by_id = {r.window_id: i for i, r in enumerate(ds.records)}
        missing = [w for w in args.window_id if w not in by_id]
        if missing:
            raise SystemExit(f"window ids not in the {args.split} manifest: {missing}")
        indices = [by_id[w] for w in args.window_id]
    else:
        indices = sorted(
            random.Random(args.seed).sample(
                range(len(ds)), min(args.num_windows, len(ds))
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        record = ds.records[idx]
        item = ds[idx]
        mix = item["speech"].mean(dim=0).numpy()
        sf.write(str(args.out_dir / f"{record.window_id}_mix.wav"), mix, ds.fs)

        branches = build_branch_texts(record.turns, record.num_channels)
        lines = [
            f"window_id:  {record.window_id}",
            f"session_id: {record.session_id}",
            f"t0={record.t0:.3f}s t1={record.t1:.3f}s "
            f"duration={record.duration:.3f}s channels={record.num_channels}",
            "",
            "branch texts (| = <turn>, # = <OTHER>):",
        ]
        lines += [f"  branch {i}: {render_tokens(b)}" for i, b in enumerate(branches)]
        lines += ["", "turns (channel, start, end, text):"]
        lines += [
            f"  ch{t.channel}  {t.start:8.3f} {t.end:8.3f}  {t.text}"
            for t in record.turns
        ]
        (args.out_dir / f"{record.window_id}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(f"wrote {record.window_id} ({record.duration:.1f}s)")
    print(f"{len(indices)} windows dumped to {args.out_dir}")


if __name__ == "__main__":
    main()
