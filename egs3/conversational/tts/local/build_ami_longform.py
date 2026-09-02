"""Build the AMI long-form external test set: one meeting = one dialogue.

Per meeting (from ``data/manifest/ami_test_sessions.jsonl``, the FULL
annotation on all four headsets):

* prompt per participant = the FIRST turn that has ``--min-words`` tokens,
  lies inside the ``--band`` duration window, is solo by annotation under
  ``--solo-guard``, and is not in the prompt gate's excluded spans; cut from
  the headset, normalized to ``--normalize-db`` active RMS;
* script = every turn in time order MINUS the prompt turns (the model never
  re-speaks its prompt); a participant with no gate-passing prompt, or with
  no turn left after removing it, is dropped from the record (reported);
* ground truth per channel = the full-length headset channel masked to that
  channel's SCRIPT turns (+``--mask-guard``; the prompt turn is masked out
  too, so the anchor's transcript and audio agree), normalized like the
  prompts, stored as FLAC.

Output (``--out``): ``manifest.jsonl`` in the training-style external
manifest shape ``src/external_testset.load_external_manifest`` reads
(``window_id``, ``num_channels``, ``turns[{channel, speaker, text}]``,
``channels[{prompt_wav, prompt_text, gt_wav}]``; extra provenance keys are
ignored by the loader), ``prompt/``, ``gt/``, ``build_meta.json``.

Design note: "Design - Beyond Two Speakers Evaluation on AMI" (long-form
arm, 2026-09-02).  Runs on a CPU node (one 4-channel meeting is ~1 GB in
memory); minutes per meeting.

Usage:
    python local/build_ami_longform.py --sessions data/manifest/ami_test_sessions.jsonl \\
        --dataset-root /work/hdd/bbjs/ttrachu/dataset/ami --exclude-spans exp/ami/gate/exclude_spans.json \\
        --out downloads/ami-longform-v1 [--meetings EN2002a ES2004a ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn  # noqa: E402
from egs3.conversational.tts.local.build_zipvoice_dialog_testset import (  # noqa: E402
    _normalized,
)
from egs3.conversational.tts.src.inference import load_excluded_spans  # noqa: E402


def _solo(turns: Sequence[Turn], turn: Turn, guard: float) -> bool:
    s, e = turn.start - guard, turn.end + guard
    return not any(o.channel != turn.channel and o.start < e and s < o.end for o in turns)


def pick_prompt_turn(
    turns: Sequence[Turn],
    channel: int,
    *,
    min_words: int,
    band: tuple[float, float],
    solo_guard: float,
    excluded: frozenset = frozenset(),
) -> Turn | None:
    """The participant's first turn that can serve as their prompt."""
    for t in sorted(turns, key=lambda t: (t.start, t.channel)):
        if t.channel != channel:
            continue
        if len(t.text.split()) < min_words:
            continue
        dur = t.end - t.start
        if not (band[0] - 1e-6 <= dur <= band[1] + 1e-6):
            continue
        if (t.channel, round(t.start, 6), round(t.end, 6)) in excluded:
            continue
        if not _solo(turns, t, solo_guard):
            continue
        return t
    return None


def _mask(channel_audio: np.ndarray, turns: Sequence[Turn], sr: int, guard: float) -> np.ndarray:
    out = np.zeros_like(channel_audio)
    n = channel_audio.shape[0]
    for t in turns:
        a = max(0, int(round((t.start - guard) * sr)))
        b = min(n, int(round((t.end + guard) * sr)))
        if b > a:
            out[a:b] = channel_audio[a:b]
    return out


def build_longform(
    *,
    sessions,
    dataset_root,
    out_dir,
    exclude_spans,
    min_words: int,
    band: tuple[float, float],
    solo_guard: float,
    mask_guard: float,
    normalize_db: float | None,
    meetings: Sequence[str] | None,
) -> dict:
    out_dir = Path(out_dir).resolve()
    (out_dir / "prompt").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    excluded_by_session = load_excluded_spans(exclude_spans) if exclude_spans else {}
    wanted = set(meetings) if meetings else None
    rows: list[dict] = []
    meta_meetings: dict[str, dict] = {}
    limited: list[str] = []
    for line in Path(sessions).read_text("utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        mid = s["session_id"]
        if wanted is not None and mid not in wanted:
            continue
        turns = [
            Turn(int(t["channel"]), t["speaker"], t["text"], float(t["start"]), float(t["end"]))
            for t in s["turns"]
        ]
        excluded = excluded_by_session.get(mid, frozenset())
        prompts: dict[int, Turn] = {}
        for ch in range(int(s["num_channels"])):
            p = pick_prompt_turn(
                turns, ch, min_words=min_words, band=band, solo_guard=solo_guard, excluded=excluded
            )
            if p is not None:
                prompts[ch] = p
        prompt_keys = {(t.channel, t.start, t.end) for t in prompts.values()}
        script = [t for t in sorted(turns, key=lambda t: (t.start, t.channel))
                  if (t.channel, t.start, t.end) not in prompt_keys]
        keep = sorted(ch for ch in prompts if any(t.channel == ch for t in script))
        dropped = sorted(set(range(int(s["num_channels"]))) - set(keep))
        if len(keep) < 1:
            meta_meetings[mid] = {"skipped": "no participant with a prompt and a script"}
            continue
        row_of = {ch: i for i, ch in enumerate(keep)}
        script = [t for t in script if t.channel in keep]

        audio, sr = sf.read(str(Path(dataset_root) / s["audio_relpath"]), dtype="float32", always_2d=True)
        chans: list[dict] = []
        for ch in keep:
            row = row_of[ch]
            p = prompts[ch]
            mono = audio[int(round(p.start * sr)):int(round(p.end * sr)), ch]
            if normalize_db is not None:
                mono, _g, lim = _normalized(mono, sr, normalize_db)
                if lim:
                    limited.append(f"{mid}_ch{row}_prompt")
            prel = f"prompt/{mid}_ch{row}.wav"
            sf.write(str(out_dir / prel), mono.astype(np.float32), sr, subtype="PCM_16")
            gt = _mask(audio[:, ch], [t for t in script if t.channel == ch], sr, mask_guard)
            if normalize_db is not None:
                gt, _g, lim = _normalized(gt, sr, normalize_db)
                if lim:
                    limited.append(f"{mid}_ch{row}_gt")
            grel = f"gt/{mid}_ch{row}.flac"
            sf.write(str(out_dir / grel), gt.astype(np.float32), sr, subtype="PCM_16", format="FLAC")
            chans.append({
                "prompt_wav": prel, "prompt_text": p.text, "gt_wav": grel,
                "speaker": p.speaker, "source_channel": ch,
                "prompt_span": [round(p.start, 6), round(p.end, 6)],
            })
        rows.append({
            "window_id": mid, "session_id": mid, "num_channels": len(keep),
            "source_channels": keep, "duration_sec": round(audio.shape[0] / sr, 6),
            "channels": chans,
            "turns": [{"channel": row_of[t.channel], "speaker": t.speaker, "text": t.text,
                       "start": round(t.start, 6), "end": round(t.end, 6)} for t in script],
        })
        meta_meetings[mid] = {
            "num_channels": len(keep), "dropped_channels": dropped, "n_turns": len(script),
            "duration_sec": round(audio.shape[0] / sr, 3),
            "prompts": {str(row_of[ch]): [round(prompts[ch].start, 3), round(prompts[ch].end, 3)] for ch in keep},
        }
        print(f"{mid}: K={len(keep)} turns={len(script)} dropped={dropped}", flush=True)
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    meta = {
        "params": {"min_words": min_words, "band": list(band), "solo_guard": solo_guard,
                   "mask_guard": mask_guard, "normalize_db": normalize_db,
                   "exclude_spans": str(exclude_spans) if exclude_spans else None,
                   "sessions": str(sessions)},
        "n_meetings": len(rows), "meetings": meta_meetings, "peak_limited": limited,
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-spans", type=Path, default=None)
    ap.add_argument("--meetings", nargs="*", default=None)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--band", type=float, nargs=2, default=(2.0, 10.0))
    ap.add_argument("--solo-guard", type=float, default=0.3)
    ap.add_argument("--mask-guard", type=float, default=0.15)
    ap.add_argument("--normalize-db", type=float, default=-23.0)
    a = ap.parse_args(argv)
    meta = build_longform(
        sessions=a.sessions, dataset_root=a.dataset_root, out_dir=a.out, exclude_spans=a.exclude_spans,
        min_words=a.min_words, band=tuple(a.band), solo_guard=a.solo_guard, mask_guard=a.mask_guard,
        normalize_db=a.normalize_db, meetings=a.meetings,
    )
    print(json.dumps({"n_meetings": meta["n_meetings"], "peak_limited": meta["peak_limited"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
