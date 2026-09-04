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

Any ``SessionRecord`` manifest works, not only AMI: the Fisher long-form arm
("Design - Long-Form Two-Speaker Evaluation on Fisher", 2026-09-04) feeds
``sessions_fisher_test.jsonl`` through the same rules.  A record's own
``exclusion_spans`` (``[[start, end], ...]`` in seconds, no channel: time the
transcript does not cover, e.g. Fisher's unintelligible utterances) are
honoured on EVERY channel - a turn overlapping a span is never a prompt
candidate and is dropped from the script, so the masked ground truth omits
it too.  ``--ids-file`` pins a subset by session id (one per line, ``#``
comments allowed), mutually exclusive with ``--meetings``.

``--prompt-pool`` switches the prompt source to EXTERNAL utterances (the
CoVoMix2 protocol; the Fisher long-form arm's PRIMARY construction): a
one-channel training-style manifest (e.g. ``downloads/libritts-test-clean``)
supplies the pool, each call draws K distinct pool speakers with one
in-band utterance each (``--pool-band``, seeded per session by
``--prompt-seed``), gender-matched when BOTH ``--pool-genders``
(LibriTTS ``SPEAKERS.txt``) and ``--channel-genders`` (json
``{session_id: [gender per channel]}``) are given and random otherwise.  No
corpus turn is spent as a prompt, so the script is the whole session and the
ground truth is complete; the assignment is written to ``build_meta.json``.

Usage:
    python local/build_ami_longform.py --sessions data/manifest/ami_test_sessions.jsonl \\
        --dataset-root /work/hdd/bbjs/ttrachu/dataset/ami --exclude-spans exp/ami/gate/exclude_spans.json \\
        --out downloads/ami-longform-v1 [--meetings EN2002a ES2004a ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
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


def _overlaps_span(turn: Turn, spans: Sequence[tuple[float, float]]) -> bool:
    return any(turn.start < e and s < turn.end for s, e in spans)


def _solo(turns: Sequence[Turn], turn: Turn, guard: float) -> bool:
    s, e = turn.start - guard, turn.end + guard
    return not any(o.channel != turn.channel and o.start < e and s < o.end for o in turns)


PROMPT_TIERS = ("gated_solo", "solo", "in_band")


def pick_prompt_turn(
    turns: Sequence[Turn],
    channel: int,
    *,
    min_words: int,
    band: tuple[float, float],
    solo_guard: float,
    excluded: frozenset = frozenset(),
    exclusion_spans: Sequence[tuple[float, float]] = (),
) -> tuple[Turn, str] | None:
    """The participant's first turn that can serve as their prompt, with the
    ladder tier it came from: ``gated_solo`` (not gate-excluded AND solo by
    annotation), then ``solo`` (gate ignored), then ``in_band`` (any lexical
    turn inside the duration band).  The band and the lexical rule are
    never relaxed, and neither is ``exclusion_spans`` (record-level time
    spans the transcript does not cover).  A participant is dropped only
    when even the last tier is empty - a speaker with a worse prompt beats
    a missing speaker in a whole-meeting generation."""
    ordered = [
        t for t in sorted(turns, key=lambda t: (t.start, t.channel))
        if t.channel == channel
        and len(t.text.split()) >= min_words
        and band[0] - 1e-6 <= t.end - t.start <= band[1] + 1e-6
        and not _overlaps_span(t, exclusion_spans)
    ]
    for tier in PROMPT_TIERS:
        for t in ordered:
            gated = (t.channel, round(t.start, 6), round(t.end, 6)) in excluded
            if tier == "gated_solo" and (gated or not _solo(turns, t, solo_guard)):
                continue
            if tier == "solo" and not _solo(turns, t, solo_guard):
                continue
            return t, tier
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


@dataclass(frozen=True)
class PoolItem:
    speaker: str
    wav: Path
    text: str
    source_id: str
    duration: float


def load_prompt_pool(manifest, band: tuple[float, float]) -> list[PoolItem]:
    """One-channel external manifest -> in-band pool items (utterance audio =
    ``channels[0].gt_wav``, transcript = ``turns[0].text``).  Paths resolve
    against the manifest directory; durations come from the audio headers."""
    manifest = Path(manifest)
    items: list[PoolItem] = []
    for line in manifest.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["num_channels"]) != 1 or len(r["turns"]) != 1:
            raise ValueError(f"{manifest}: pool rows must be one-channel single-turn, got {r['window_id']}")
        ch = r["channels"][0]
        wav = Path(ch["gt_wav"])
        if not wav.is_absolute():
            wav = manifest.parent / wav
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate
        if band[0] - 1e-6 <= dur <= band[1] + 1e-6:
            items.append(PoolItem(str(ch.get("speaker") or r["turns"][0]["speaker"]), wav,
                                  r["turns"][0]["text"], r["window_id"], dur))
    if not items:
        raise ValueError(f"{manifest}: no pool utterance inside the {band} s band")
    return items


def load_pool_genders(path) -> dict[str, str]:
    """LibriTTS/LibriSpeech ``SPEAKERS.txt``: ``ID | SEX | SUBSET | ...`` -> {id: sex}."""
    out: dict[str, str] = {}
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 2:
            out[cols[0]] = cols[1].upper()
    return out


def draw_prompts(
    pool: Sequence[PoolItem],
    session_id: str,
    num_channels: int,
    seed: int,
    *,
    channel_genders: Sequence[str] | None = None,
    pool_genders: dict[str, str] | None = None,
) -> list[PoolItem]:
    """K distinct pool speakers for one session, one utterance each, seeded
    by ``f"{seed}:{session_id}"`` (stable under pool growth of OTHER
    speakers only if the speaker list keeps its order).  Gender-matched per
    channel when both gender sources are given and the matching gender has a
    speaker left; otherwise the draw is over every remaining speaker."""
    rng = random.Random(f"{seed}:{session_id}")
    by_spk: dict[str, list[PoolItem]] = {}
    for it in pool:
        by_spk.setdefault(it.speaker, []).append(it)
    remaining = sorted(by_spk)
    if len(remaining) < num_channels:
        raise ValueError(f"{session_id}: pool has {len(remaining)} speakers, need {num_channels}")
    chosen: list[PoolItem] = []
    for ch in range(num_channels):
        cands = remaining
        if channel_genders is not None and pool_genders is not None:
            want = str(channel_genders[ch]).upper()
            matched = [s for s in remaining if pool_genders.get(s, "").upper() == want]
            if matched:
                cands = matched
        spk = rng.choice(cands)
        remaining = [s for s in remaining if s != spk]
        chosen.append(rng.choice(sorted(by_spk[spk], key=lambda it: it.source_id)))
    return chosen


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
    ids_file=None,
    prompt_pool=None,
    pool_band: tuple[float, float] = (3.0, 10.0),
    prompt_seed: int = 0,
    pool_genders=None,
    channel_genders=None,
) -> dict:
    out_dir = Path(out_dir).resolve()
    if ids_file is not None:
        if meetings:
            raise ValueError("ids_file and meetings are mutually exclusive")
        meetings = [
            ln.strip() for ln in Path(ids_file).read_text("utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if not meetings:
            raise ValueError(f"{ids_file}: no session ids")
    sessions_md5 = hashlib.md5(Path(sessions).read_bytes()).hexdigest()
    pool = load_prompt_pool(prompt_pool, pool_band) if prompt_pool is not None else None
    pool_gender_map = load_pool_genders(pool_genders) if (pool is not None and pool_genders) else None
    chan_gender_map = (
        json.loads(Path(channel_genders).read_text("utf-8"))
        if (pool is not None and channel_genders) else None
    )
    (out_dir / "prompt").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    excluded_by_session = load_excluded_spans(exclude_spans) if exclude_spans else {}
    wanted = set(meetings) if meetings else None
    rows: list[dict] = []
    meta_meetings: dict[str, dict] = {}
    limited: list[str] = []
    n_records_with_spans = 0
    n_turns_excluded_total = 0
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
        spans = [(float(a), float(b)) for a, b in (s.get("exclusion_spans") or [])]
        if spans:
            n_records_with_spans += 1
        prompts: dict[int, Turn] = {}
        tiers: dict[int, str] = {}
        external: dict[int, PoolItem] = {}
        if pool is not None:
            genders = chan_gender_map.get(mid) if chan_gender_map else None
            if chan_gender_map is not None and genders is None:
                raise ValueError(f"{mid}: no entry in channel_genders")
            drawn = draw_prompts(pool, mid, int(s["num_channels"]), prompt_seed,
                                 channel_genders=genders, pool_genders=pool_gender_map)
            for ch, it in enumerate(drawn):
                external[ch] = it
                tiers[ch] = "external"
        else:
            for ch in range(int(s["num_channels"])):
                picked = pick_prompt_turn(
                    turns, ch, min_words=min_words, band=band, solo_guard=solo_guard,
                    excluded=excluded, exclusion_spans=spans,
                )
                if picked is not None:
                    prompts[ch], tiers[ch] = picked
        prompt_keys = {(t.channel, t.start, t.end) for t in prompts.values()}
        ordered_turns = sorted(turns, key=lambda t: (t.start, t.channel))
        n_turns_excluded = sum(1 for t in ordered_turns if _overlaps_span(t, spans))
        n_turns_excluded_total += n_turns_excluded
        script = [t for t in ordered_turns
                  if (t.channel, t.start, t.end) not in prompt_keys and not _overlaps_span(t, spans)]
        keep = sorted(ch for ch in (external if pool is not None else prompts)
                      if any(t.channel == ch for t in script))
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
            if pool is not None:
                it = external[ch]
                mono, psr = sf.read(str(it.wav), dtype="float32", always_2d=True)
                if psr != sr:
                    raise ValueError(f"{mid}: pool utterance {it.source_id} is {psr} Hz, session is {sr} Hz")
                mono = mono[:, 0]
                p_text, p_speaker, p_span, p_source = it.text, it.speaker, None, it.source_id
            else:
                p = prompts[ch]
                mono = audio[int(round(p.start * sr)):int(round(p.end * sr)), ch]
                p_text, p_speaker, p_span, p_source = p.text, p.speaker, [round(p.start, 6), round(p.end, 6)], None
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
                "prompt_wav": prel, "prompt_text": p_text, "gt_wav": grel,
                "speaker": p_speaker, "source_channel": ch,
                "prompt_span": p_span, "prompt_source": p_source,
                "prompt_tier": tiers[ch],
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
            "prompts": ({str(row_of[ch]): external[ch].source_id for ch in keep} if pool is not None
                        else {str(row_of[ch]): [round(prompts[ch].start, 3), round(prompts[ch].end, 3)] for ch in keep}),
            "prompt_speakers": {str(row_of[ch]): (external[ch].speaker if pool is not None else prompts[ch].speaker)
                                for ch in keep},
            "prompt_tiers": {str(row_of[ch]): tiers[ch] for ch in keep},
            "n_turns_excluded": n_turns_excluded,
            "exclusion_sec": round(sum(b - a for a, b in spans), 3),
        }
        print(
            f"{mid}: K={len(keep)} turns={len(script)} dropped={dropped} "
            f"tiers={[tiers[ch] for ch in keep]} excluded_turns={n_turns_excluded}",
            flush=True,
        )
    (out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    meta = {
        "params": {"min_words": min_words, "band": list(band), "solo_guard": solo_guard,
                   "mask_guard": mask_guard, "normalize_db": normalize_db,
                   "exclude_spans": str(exclude_spans) if exclude_spans else None,
                   "sessions": str(sessions), "ids_file": str(ids_file) if ids_file else None,
                   "prompt_pool": str(prompt_pool) if prompt_pool else None,
                   "pool_band": list(pool_band), "prompt_seed": prompt_seed,
                   "pool_genders": str(pool_genders) if pool_genders else None,
                   "channel_genders": str(channel_genders) if channel_genders else None},
        "pool_size": (len(pool) if pool is not None else None),
        "pool_speakers": (len({it.speaker for it in pool}) if pool is not None else None),
        "sessions_md5": sessions_md5,
        "exclusion": {"n_records_with_spans": n_records_with_spans,
                      "n_turns_excluded": n_turns_excluded_total},
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
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--meetings", nargs="*", default=None, help="session ids to build (default: all)")
    grp.add_argument("--ids-file", type=Path, default=None,
                     help="file of session ids, one per line (# comments allowed); pins a subset")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--band", type=float, nargs=2, default=(2.0, 10.0))
    ap.add_argument("--solo-guard", type=float, default=0.3)
    ap.add_argument("--mask-guard", type=float, default=0.15)
    ap.add_argument("--normalize-db", type=float, default=-23.0)
    ap.add_argument("--prompt-pool", type=Path, default=None,
                    help="one-channel external manifest of prompt utterances (CoVoMix2-style external prompts)")
    ap.add_argument("--pool-band", type=float, nargs=2, default=(3.0, 10.0))
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--pool-genders", type=Path, default=None, help="SPEAKERS.txt of the pool corpus")
    ap.add_argument("--channel-genders", type=Path, default=None,
                    help="json {session_id: [gender per channel]} for gender-matched draws")
    a = ap.parse_args(argv)
    meta = build_longform(
        sessions=a.sessions, dataset_root=a.dataset_root, out_dir=a.out, exclude_spans=a.exclude_spans,
        min_words=a.min_words, band=tuple(a.band), solo_guard=a.solo_guard, mask_guard=a.mask_guard,
        normalize_db=a.normalize_db, meetings=a.meetings, ids_file=a.ids_file,
        prompt_pool=a.prompt_pool, pool_band=tuple(a.pool_band), prompt_seed=a.prompt_seed,
        pool_genders=a.pool_genders, channel_genders=a.channel_genders,
    )
    print(json.dumps({"n_meetings": meta["n_meetings"], "peak_limited": meta["peak_limited"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
