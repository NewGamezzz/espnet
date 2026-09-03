"""AMI test-partition builder: meetings -> merged turns -> windows -> K strata.

Output (relative to ``<recipe_dir>/data`` and ``<recipe_dir>``):

* ``manifest/ami_test.jsonl``          window manifest; every record has
  ``num_channels = 4``, ``channels`` = the headset indices of its K lexically
  active participants, and only THEIR turns (design note section 2).  K = 1
  windows are kept as prompt-pool material (never scored).
* ``manifest/ami_test_sessions.jsonl`` one line per meeting with the FULL
  merged turn list on all four channels - the prompt gate and the crosstalk
  table need the whole annotation, not the windowed subset.
* ``exp/ami/window_report.json``       the duration x K distribution the
  per-stratum caps are chosen from (never promise counts before this exists).

Usage (a CPU-partition job, not the login node - the espnet import alone
takes minutes on /work/hdd):

    python -m egs3.conversational.tts.dataset.ami_builder --base-vocab-path <vocab>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path
from typing import Sequence

from espnet3.utils.config_utils import load_config_with_defaults

from .preprocessing.ami import (
    TEST_MEETINGS,
    complete_participants,
    load_ami_recordings,
    load_meetings,
    load_words,
    words_to_supervisions,
)
from .preprocessing.sssd import Turn, merge_turns
from .preprocessing.text import normalize_text, vocab_charset
from .preprocessing.windows import (
    WindowRecord,
    WindowingStats,
    build_windows,
    to_json,
)


def _load_cfg() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)[
            "ami_builder"
        ]


_CFG = _load_cfg()


def lexical_active_channels(turns: Sequence[Turn], min_words: int) -> tuple[int, ...]:
    """Channels with at least one turn of ``min_words`` or more word tokens."""
    active = {t.channel for t in turns if len(t.text.split()) >= min_words}
    return tuple(sorted(active))


def stratify_window(record: WindowRecord, min_words: int) -> WindowRecord | None:
    """Reduce a 4-channel window to its K lexically active participants.

    Non-active participants' turns (backchannels, fragments) are dropped
    together with their channel: the K-channel ground truth is those K
    headsets and nothing else.  K = 1 windows (monologues) are KEPT as
    prompt-pool material only - ``selection.num_active_speakers`` never
    selects them for scoring, but their turns are the best solo prompts a
    meeting has.  Returns None only when no participant is lexically active.
    """
    channels = lexical_active_channels(record.turns, min_words)
    if not channels:
        return None
    turns = tuple(t for t in record.turns if t.channel in channels)
    if len(channels) >= 2 and len(turns) < 2:
        return None
    return dataclasses.replace(record, turns=turns, channels=channels)


def _session_line(mid: str, rec, speakers: dict[int, str], turns) -> dict:
    return {
        "session_id": mid,
        "audio_relpath": rec.audio_relpath,
        "sample_rate": rec.sample_rate,
        "num_channels": rec.num_channels,
        "duration": rec.duration,
        "speakers": {str(ch): name for ch, name in sorted(speakers.items())},
        "turns": [
            {
                "channel": t.channel,
                "speaker": t.speaker,
                "text": t.text,
                "start": round(t.start, 6),
                "end": round(t.end, 6),
            }
            for t in turns
        ],
    }


def _quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)

    def q(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    return {"min": s[0], "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "max": s[-1]}


class AMIBuilder:
    def build(
        self,
        recipe_dir: str | Path,
        dataset_root: str | Path | None = None,
        base_vocab_path: str | Path | None = None,
        meetings: Sequence[str] = TEST_MEETINGS,
        seed: int | None = None,
    ) -> dict:
        cfg = _CFG
        root = Path(dataset_root or cfg["dataset_root"])
        seed = int(seed if seed is not None else cfg["seed"])
        recipe_dir = Path(recipe_dir).resolve()
        data_dir = recipe_dir / "data"
        if base_vocab_path is None:
            raise ValueError(
                "base_vocab_path is required: the training vocab that defines "
                "the normalizer's charset (training_config.dataset.preprocessor."
                "token_list)"
            )
        tokens = Path(base_vocab_path).read_bytes().decode("utf-8").splitlines()
        charset = vocab_charset(tokens)

        ann = root / cfg["annotations_subdir"]
        participants = load_meetings(
            ann / "corpusResources" / "meetings.xml", require=list(meetings)
        )
        recordings = load_ami_recordings(root, root / cfg["flac_subdir"], meetings)

        manifest_path = data_dir / cfg["manifest_path"]
        sessions_path = data_dir / cfg["sessions_path"]
        report_path = recipe_dir / cfg["report_path"]
        for p in (manifest_path, sessions_path, report_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        stats = WindowingStats()
        per_k: dict[int, dict] = defaultdict(
            lambda: {"windows": 0, "seconds": 0.0, "per_meeting": Counter()}
        )
        durations: list[float] = []
        # Both reasons always present in the report, zero or not.
        dropped: Counter = Counter({"no_lexical_speaker": 0, "empty_after_normalize": 0})
        n_windows = 0
        synthesized_participants: dict[str, list[dict]] = {}
        with manifest_path.open("w", encoding="utf-8") as mf, sessions_path.open(
            "w", encoding="utf-8"
        ) as sfh:
            for mid in meetings:
                rec = recordings[mid]
                parts, synthesized = complete_participants(
                    mid, participants[mid], ann / "words"
                )
                if synthesized:
                    synthesized_participants[mid] = [
                        {"agent": p.agent, "channel": p.channel, "speaker": p.global_name}
                        for p in synthesized
                    ]
                sups = []
                speakers: dict[int, str] = {}
                for p in sorted(parts, key=lambda p: p.channel):
                    words = load_words(ann / "words" / f"{mid}.{p.agent}.words.xml")
                    sups.extend(
                        words_to_supervisions(
                            words,
                            meeting_id=mid,
                            channel=p.channel,
                            speaker=p.global_name,
                            utterance_gap=float(cfg["utterance_gap"]),
                            agent=p.agent,
                        )
                    )
                    speakers[p.channel] = p.global_name
                turns = merge_turns(sups, float(cfg["merge_gap"]))
                normalized = []
                for t in turns:
                    text = normalize_text(t.text, charset)
                    if text:
                        normalized.append(dataclasses.replace(t, text=text))
                    else:
                        dropped["empty_after_normalize"] += 1
                sfh.write(
                    json.dumps(_session_line(mid, rec, speakers, normalized), ensure_ascii=False)
                    + "\n"
                )

                records, session_stats = build_windows(
                    mid,
                    rec,
                    normalized,
                    window_min=float(cfg["window_min"]),
                    window_max=float(cfg["window_max"]),
                    boundary_guard=float(cfg["boundary_guard"]),
                    tail_min=float(cfg["tail_min"]),
                    rng=random.Random(f"{seed}:window:{mid}"),
                    trim_to_turns=bool(cfg["trim_to_turns"]),
                    min_coverage=float(cfg["min_coverage"]),
                    snap_start_to_turn=bool(cfg["snap_start_to_turn"]),
                )
                stats.merge(session_stats)
                for r in records:
                    strat = stratify_window(r, int(cfg["min_words"]))
                    if strat is None:
                        dropped["no_lexical_speaker"] += 1
                        continue
                    mf.write(json.dumps(to_json(strat), ensure_ascii=False) + "\n")
                    k = strat.num_rows
                    per_k[k]["windows"] += 1
                    per_k[k]["seconds"] += strat.duration
                    per_k[k]["per_meeting"][mid] += 1
                    durations.append(strat.duration)
                    n_windows += 1

        windowing = dataclasses.asdict(stats)
        windowing["cut_gaps"] = len(stats.cut_gaps)
        report = {
            "meetings": list(meetings),
            "config": {
                k: cfg[k]
                for k in (
                    "window_min",
                    "window_max",
                    "tail_min",
                    "boundary_guard",
                    "trim_to_turns",
                    "snap_start_to_turn",
                    "merge_gap",
                    "utterance_gap",
                    "min_words",
                    "seed",
                )
            },
            "n_windows": n_windows,
            "per_k": {
                str(k): {
                    "windows": v["windows"],
                    "minutes": round(v["seconds"] / 60, 2),
                    "per_meeting": dict(v["per_meeting"]),
                }
                for k, v in sorted(per_k.items())
            },
            "duration_quantiles": _quantiles(durations),
            "dropped": dict(dropped),
            # Agents with a words file but no meetings.xml row (EN2002c agent A).
            "synthesized_participants": synthesized_participants,
            "windowing": windowing,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recipe-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--base-vocab-path", type=Path, required=True)
    ap.add_argument("--meetings", nargs="*", default=list(TEST_MEETINGS))
    args = ap.parse_args(argv)
    report = AMIBuilder().build(
        args.recipe_dir, args.dataset_root, args.base_vocab_path, args.meetings
    )
    print(
        json.dumps(
            {k: report[k] for k in ("n_windows", "per_k", "duration_quantiles", "dropped")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
