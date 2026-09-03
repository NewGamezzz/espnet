"""Prompt gate and bleed table for the AMI test partition.

For every meeting in ``data/manifest/ami_test_sessions.jsonl`` (the FULL
annotation, all four headsets):

1. per-channel silence FLOOR = RMS (dBFS) over regions where nobody speaks
   (annotation, shrunk by ``--silence-guard``);
2. every turn of at least ``--min-candidate-sec`` is a prompt candidate; it is
   EXCLUDED when (a) another participant has a word inside the span widened
   by ``--solo-guard`` on each side ("not_solo"), or (b) some other headset's
   RMS over the span exceeds ITS OWN floor by more than ``--max-excess-db``
   ("energy:ch<k>").  Gating against each channel's own floor, not against
   the prompt channel, is deliberate: AMI headset gains are unbalanced;
3. the bleed table of ``local/crosstalk_report.py`` is reproduced for these
   meetings (solo-region energy of every other channel relative to the
   speaking one) so the paper's appendix can quote it.

Outputs: ``<out-dir>/exclude_spans.json`` (consumed by ``prompt.exclude_spans``)
and ``<out-dir>/bleed_table.tsv``.  Runs on the cpu partition; seek-reads
only the measured spans.

Usage:
    python local/ami_prompt_gate.py --sessions data/manifest/ami_test_sessions.jsonl \\
        --dataset-root /work/hdd/bbjs/ttrachu/dataset/ami --out-dir exp/ami/gate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn  # noqa: E402

Interval = tuple[float, float]
_FLOOR_DB = -120.0


def _merge(spans: Sequence[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def silence_regions(turns: Sequence[Turn], duration: float, guard: float) -> list[Interval]:
    """Regions where no participant is active, each turn widened by ``guard``."""
    busy = _merge([(t.start - guard, t.end + guard) for t in turns])
    out: list[Interval] = []
    cur = 0.0
    for s, e in busy:
        if s > cur:
            out.append((round(cur, 6), round(s, 6)))
        cur = max(cur, e)
    if cur < duration:
        out.append((round(cur, 6), round(duration, 6)))
    return out


def solo_by_annotation(turns: Sequence[Turn], turn: Turn, guard: float) -> bool:
    s, e = turn.start - guard, turn.end + guard
    return not any(o.channel != turn.channel and o.start < e and s < o.end for o in turns)


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return _FLOOR_DB
    return float(10 * math.log10(max(float(np.mean(np.square(x, dtype=np.float64))), 1e-12)))


def _read(audio, s: float, e: float, sr: int) -> np.ndarray:
    """``audio`` is an in-memory ``(T, C)`` array or an open ``SoundFile``.
    Bounds are clamped to the file: annotation times can round a hair past
    the last frame (TS3007c), and libsndfile refuses to seek there."""
    total = audio.shape[0] if isinstance(audio, np.ndarray) else audio.frames
    a = min(max(0, int(round(s * sr))), total)
    b = min(max(a, int(round(e * sr))), total)
    if isinstance(audio, np.ndarray):
        return audio[a:b]
    if b <= a:
        return np.zeros((0, audio.channels), dtype="float32")
    audio.seek(a)
    return audio.read(b - a, dtype="float32", always_2d=True)


def channel_floor_db(audio, regions: Sequence[Interval], sr: int) -> list[float]:
    """Per-channel RMS dBFS pooled over ``regions`` (energy-weighted by length)."""
    sumsq = None
    frames = 0
    for s, e in regions:
        block = _read(audio, s, e, sr)
        if block.shape[0] == 0:
            continue
        sq = np.square(block, dtype=np.float64).sum(axis=0)
        sumsq = sq if sumsq is None else sumsq + sq
        frames += block.shape[0]
    if sumsq is None or frames == 0:
        raise ValueError("no silent region to estimate the floor from")
    return [float(10 * math.log10(max(v / frames, 1e-12))) for v in sumsq]


def turn_excess_db(audio, turn: Turn, sr: int, floors: Sequence[float]) -> list[float]:
    """Per channel: RMS over the turn span minus that channel's own floor."""
    block = _read(audio, turn.start, turn.end, sr)
    return [_rms_db(block[:, k]) - floors[k] for k in range(block.shape[1])]


def gate_session(
    session_id: str,
    audio_path,
    turns: Sequence[Turn],
    *,
    duration: float,
    turn_min: float,
    turn_max: float,
    solo_guard: float,
    max_excess_db: float,
    silence_guard: float,
    min_candidate_sec: float = 1.0,
) -> dict:
    """Gate every candidate turn of one session; see the module docstring.

    ``turn_min``/``turn_max`` are recorded for provenance; every turn at
    least ``min_candidate_sec`` long is evaluated so the ladder's relaxed
    (out-of-band) tiers stay gated too.
    """
    with sf.SoundFile(str(audio_path)) as audio:
        sr = audio.samplerate
        floors = channel_floor_db(audio, silence_regions(turns, duration, silence_guard), sr)
        excluded: list[dict] = []
        n_cand = n_acc = 0
        excess_log: list[float] = []
        solo_worst: list[float] = []  # every solo candidate, before the margin
        for t in turns:
            if t.end - t.start < min_candidate_sec:
                continue
            n_cand += 1
            entry = {
                "session_id": session_id,
                "channel": int(t.channel),
                "start": round(float(t.start), 6),
                "end": round(float(t.end), 6),
            }
            if not solo_by_annotation(turns, t, solo_guard):
                excluded.append({**entry, "reason": "not_solo"})
                continue
            excess = turn_excess_db(audio, t, sr, floors)
            others = [(k, x) for k, x in enumerate(excess) if k != t.channel]
            worst_k, worst = max(others, key=lambda kx: kx[1])
            solo_worst.append(round(worst, 3))
            if worst > max_excess_db:
                excluded.append({**entry, "reason": f"energy:ch{worst_k}"})
                continue
            n_acc += 1
            excess_log.append(round(worst, 3))
    return {
        "session_id": session_id,
        "floor_db": floors,
        "band": [turn_min, turn_max],
        "excluded": excluded,
        "n_candidates": n_cand,
        "n_not_solo": sum(1 for e in excluded if e["reason"] == "not_solo"),
        "n_energy": sum(1 for e in excluded if e["reason"].startswith("energy")),
        "n_accepted": n_acc,
        "accepted_worst_excess_db": excess_log,
        "solo_worst_excess_db": solo_worst,
    }


def bleed_rows(session_id: str, audio_path, turns: Sequence[Turn], *, guard: float = 0.2) -> list[tuple]:
    """``crosstalk_report.py``'s measure on one session: for every speaking
    channel j, the energy of every other channel k over j's solo regions,
    relative to j's own energy there.  Row = (session_id, k, j, solo_sec,
    bleed_db, own_db)."""
    per: dict[int, list[Interval]] = {}
    for t in turns:
        per.setdefault(int(t.channel), []).append((t.start, t.end))
    per = {c: _merge(v) for c, v in per.items()}
    rows: list[tuple] = []
    with sf.SoundFile(str(audio_path)) as audio:
        sr = audio.samplerate
        for j, own in sorted(per.items()):
            others = _merge([iv for c, v in per.items() if c != j for iv in v])
            solo: list[Interval] = []
            for s, e in own:
                cur = s
                for os_, oe in others:
                    if oe <= cur or os_ >= e:
                        continue
                    if os_ > cur:
                        solo.append((cur, os_))
                    cur = max(cur, oe)
                if cur < e:
                    solo.append((cur, e))
            solo = [(s + guard, e - guard) for s, e in solo if e - s > 2 * guard]
            solo_sec = sum(e - s for s, e in solo)
            if solo_sec < 1.0:
                continue
            sumsq = None
            frames = 0
            for s, e in solo:
                block = _read(audio, s, e, sr)
                sq = np.square(block, dtype=np.float64).sum(axis=0)
                sumsq = sq if sumsq is None else sumsq + sq
                frames += block.shape[0]
            power = sumsq / max(frames, 1)
            own_db = 10 * math.log10(max(power[j], 1e-12))
            for k in range(len(power)):
                if k == j:
                    continue
                bleed = 10 * math.log10(max(power[k], 1e-12) / max(power[j], 1e-12))
                rows.append((session_id, k, j, round(solo_sec, 1), round(bleed, 2), round(own_db, 2)))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--turn-min", type=float, default=2.0)
    ap.add_argument("--turn-max", type=float, default=10.0)
    ap.add_argument("--solo-guard", type=float, default=0.3)
    ap.add_argument("--max-excess-db", type=float, default=6.0)
    ap.add_argument("--silence-guard", type=float, default=0.5)
    ap.add_argument("--min-candidate-sec", type=float, default=1.0)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spans: list[dict] = []
    table: list[tuple] = []
    floors: dict[str, list[float]] = {}
    worst: list[float] = []
    per_session: dict[str, dict] = {}
    for line in args.sessions.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        turns = [
            Turn(int(t["channel"]), t["speaker"], t["text"], float(t["start"]), float(t["end"]))
            for t in s["turns"]
        ]
        path = args.dataset_root / s["audio_relpath"]
        res = gate_session(
            s["session_id"], path, turns,
            duration=float(s["duration"]), turn_min=args.turn_min, turn_max=args.turn_max,
            solo_guard=args.solo_guard, max_excess_db=args.max_excess_db,
            silence_guard=args.silence_guard, min_candidate_sec=args.min_candidate_sec,
        )
        spans.extend(res["excluded"])
        floors[s["session_id"]] = res["floor_db"]
        worst.extend(res["accepted_worst_excess_db"])
        sw = sorted(res["solo_worst_excess_db"])
        q = (lambda p: sw[min(len(sw) - 1, int(round(p * (len(sw) - 1))))]) if sw else (lambda p: None)
        per_session[s["session_id"]] = {
            "n_candidates": res["n_candidates"], "n_not_solo": res["n_not_solo"],
            "n_energy": res["n_energy"], "n_accepted": res["n_accepted"],
            "solo_worst_excess_p50": q(0.5), "solo_worst_excess_p90": q(0.9),
            "accept_at_margin": {str(m): sum(1 for x in sw if x <= m) for m in (6, 8, 10, 12, 15)},
        }
        table.extend(bleed_rows(s["session_id"], path, turns))
        print(
            f"{s['session_id']}: {res['n_accepted']}/{res['n_candidates']} accepted "
            f"(not_solo {res['n_not_solo']}, energy {res['n_energy']}); solo candidates {len(sw)}, "
            f"worst-excess p50 {q(0.5)} p90 {q(0.9)} dB; accept@6/8/10/12/15 dB = "
            f"{[sum(1 for x in sw if x <= m) for m in (6, 8, 10, 12, 15)]}; "
            f"floor dB {[round(f, 1) for f in res['floor_db']]}",
            flush=True,
        )
    (args.out_dir / "exclude_spans.json").write_text(
        json.dumps(
            {
                "version": 1,
                "params": {
                    k: getattr(args, k)
                    for k in ("turn_min", "turn_max", "solo_guard", "max_excess_db",
                              "silence_guard", "min_candidate_sec")
                },
                "floor_db": floors,
                "per_session": per_session,
                "spans": spans,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    with (args.out_dir / "bleed_table.tsv").open("w", encoding="utf-8") as f:
        f.write("session_id\tchannel\tonly_other_channel\tonly_other_sec\tbleed_db\town_db\n")
        for r in table:
            f.write("\t".join(str(x) for x in r) + "\n")
    if worst:
        w = sorted(worst)
        print(
            f"accepted candidates: {len(w)}; worst-other-channel excess over floor "
            f"p50 {w[len(w) // 2]:.1f} dB, p90 {w[int(0.9 * (len(w) - 1))]:.1f} dB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
