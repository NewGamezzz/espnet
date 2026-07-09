"""Measure channel bleed on a sample of SSSD sessions.

Channel purity is expected (participants recorded on their own devices) but
not verified; this report measures it.  For every ordered channel pair
(k, j != k) it finds "solo-j" regions where, per the supervisions, ONLY
channel j is active (shrunk by a guard at the edges), then compares channel
k's energy there against channel j's own solo-speech energy:

    bleed_db(k <- j) = 10 * log10(E[ch_k^2 | solo j] / E[ch_j^2 | solo j])

This is a report, not an assertion: the numbers decide later whether
crosstalk suppression is needed.  Runs straight off the corpus manifests
(no build required) and seek-reads only the measured regions.

Usage:
    python local/crosstalk_report.py --num-sessions 20 --out exp/crosstalk_report.tsv
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from egs3.conversational.tts.dataset.builder import (  # noqa: E402
    _CFG,
    resolve_dataset_root,
)
from egs3.conversational.tts.dataset.sssd import (  # noqa: E402
    load_recordings,
    load_supervisions,
)

Interval = tuple[float, float]


def merge_intervals(spans: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(base: list[Interval], cut: list[Interval]) -> list[Interval]:
    """base minus cut; both sorted and non-overlapping."""
    out: list[Interval] = []
    for b0, b1 in base:
        cur = b0
        for c0, c1 in cut:
            if c1 <= cur or c0 >= b1:
                continue
            if c0 > cur:
                out.append((cur, c0))
            cur = max(cur, c1)
            if cur >= b1:
                break
        if cur < b1:
            out.append((cur, b1))
    return out


def solo_regions(
    per_channel: dict[int, list[Interval]], channel: int, guard: float
) -> list[Interval]:
    """Regions where only ``channel`` is active, shrunk by ``guard`` per side."""
    others = merge_intervals(
        [iv for c, ivs in per_channel.items() if c != channel for iv in ivs]
    )
    solo = subtract_intervals(per_channel[channel], others)
    return [(s + guard, e - guard) for s, e in solo if e - s > 2 * guard]


def region_sumsq(
    audio: sf.SoundFile, regions: list[Interval], sr: int
) -> tuple[np.ndarray, int]:
    """Per-channel sum of squares and total frame count over the regions."""
    sumsq = np.zeros(audio.channels, dtype=np.float64)
    frames = 0
    for start_s, end_s in regions:
        start, stop = round(start_s * sr), round(end_s * sr)
        stop = min(stop, audio.frames)
        if stop <= start:
            continue
        audio.seek(start)
        block = audio.read(stop - start, dtype="float32", always_2d=True)
        sumsq += (block.astype(np.float64) ** 2).sum(axis=0)
        frames += block.shape[0]
    return sumsq, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sessions", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("exp/crosstalk_report.tsv"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--guard",
        type=float,
        default=0.2,
        help="seconds trimmed from each solo-region edge against loose alignments",
    )
    args = parser.parse_args()

    root = resolve_dataset_root(args.dataset_root)
    manifests = root / _CFG["manifests_subdir"]
    recordings = load_recordings(
        manifests / "recordings.jsonl.gz", audio_subdir=_CFG["audio_subdir"]
    )
    supervisions = load_supervisions(manifests / "supervisions.jsonl.gz", recordings)
    session_ids = sorted(set(recordings) & set(supervisions))
    sample = sorted(
        random.Random(args.seed).sample(
            session_ids, min(args.num_sessions, len(session_ids))
        )
    )

    rows: list[tuple] = []
    for sid in sample:
        rec = recordings[sid]
        per_channel: dict[int, list[Interval]] = {
            c: [] for c in range(rec.num_channels)
        }
        for sup in supervisions[sid]:
            per_channel.setdefault(sup.channel, []).append((sup.start, sup.end))
        per_channel = {c: merge_intervals(ivs) for c, ivs in per_channel.items()}

        with sf.SoundFile(str(root / rec.audio_relpath)) as audio:
            sr = audio.samplerate
            for j in range(rec.num_channels):
                regions = solo_regions(per_channel, j, args.guard)
                solo_sec = sum(e - s for s, e in regions)
                if solo_sec < 1.0:
                    continue
                sumsq, frames = region_sumsq(audio, regions, sr)
                power = sumsq / max(frames, 1)
                own_db = 10 * math.log10(max(power[j], 1e-12))
                for k in range(rec.num_channels):
                    if k == j:
                        continue
                    bleed_db = 10 * math.log10(
                        max(power[k], 1e-12) / max(power[j], 1e-12)
                    )
                    rows.append((sid, k, j, solo_sec, bleed_db, own_db))
        print(f"measured {sid}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            "session_id\tchannel\tonly_other_channel\t"
            "only_other_sec\tbleed_db\town_db\n"
        )
        for sid, k, j, sec, bleed, own in rows:
            f.write(f"{sid}\t{k}\t{j}\t{sec:.1f}\t{bleed:.2f}\t{own:.2f}\n")

    if not rows:
        print("no measurable solo regions found")
        return
    bleeds = sorted(r[4] for r in rows)
    p90 = bleeds[min(len(bleeds) - 1, round(0.9 * (len(bleeds) - 1)))]
    print(f"\ncrosstalk over {len(sample)} sessions ({len(rows)} channel pairs):")
    print(
        f"  bleed dB (ch_k energy while only ch_j speaks, relative to ch_j): "
        f"mean={statistics.fmean(bleeds):.2f} median={statistics.median(bleeds):.2f} "
        f"p90={p90:.2f} max={bleeds[-1]:.2f}"
    )
    worst = sorted(rows, key=lambda r: -r[4])[:5]
    print("  worst pairs:")
    for sid, k, j, sec, bleed, own in worst:
        print(f"    {sid} ch{k}<-ch{j}: {bleed:.2f} dB over {sec:.0f}s solo")
    print(f"  full table: {args.out}")


if __name__ == "__main__":
    main()
