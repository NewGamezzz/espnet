"""BagPiper lm_tts dataset builder: dialogue emission CLI (Task 5).

Thin orchestration in the F5 recipe's house style (see
``egs3/conversational/tts/dataset/builder.py``): the algorithms live in
``preprocessing/`` (manifest parsing, windowing, audio, attribute
measurement) and ``emit.py`` (record assembly), so tests hit them with
fabricated fixtures; this module wires them together and writes files.

The corpus directory (``dataset_root``) is treated as strictly read-only;
every write goes under ``out_dir``.

Build outputs (under ``<out_dir>/``):
  - ``audio/<split>/<window_id>_ch{c}.wav`` / ``..._mix.wav``  cut window wavs
  - ``{tac,mono}/{train,valid,test}/dialogues.jsonl``  one JSON record per line
  - ``{tac,mono}/{train,valid,test}/dataset.json``  ``{data_entry, samples}``
    wrapper (schema: ``docs/bagpiper-findings.md``, "SFT data schema")
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path
from typing import Sequence

import yaml

from .captions import voice_description
from .emit import emit_mono_record, emit_tac_records, is_tac_eligible
from .preprocessing.attributes import audit_gender_metadata, measure_speaker
from .preprocessing.audio import cut_window_wavs, load_window_channel
from .preprocessing.sssd import Turn, load_recordings, load_supervisions, merge_turns
from .preprocessing.windows import WindowingStats, build_windows

VARIANTS = ("tac", "mono")
SPLITS = ("train", "valid", "test")

_RECIPE_DIR = Path(__file__).resolve().parents[1]


def load_config(conf_path: str | Path) -> dict:
    """Load the ``builder:`` block of ``conf/dataset.yaml`` (or an override)."""
    return yaml.safe_load(Path(conf_path).read_text())["builder"]


def resolve_dataset_root(explicit: str | Path | None, cfg: dict) -> Path:
    """Corpus root resolution: explicit argument > ``$SSSD_ROOT`` > config value.

    Mirrors the F5 recipe's ``resolve_dataset_root``.
    """
    if explicit is not None:
        return Path(explicit)
    if os.environ.get("SSSD_ROOT"):
        return Path(os.environ["SSSD_ROOT"])
    return Path(cfg["dataset_root"])


def split_sessions(
    session_ids: Sequence[str], ratios: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Seeded session-level split; test and valid sizes round first so the
    rounding never eats the small splits, train takes the rest. Identical
    algorithm to the F5 recipe's ``split_sessions``."""
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1, got {total}")
    ids = sorted(session_ids)
    random.Random(f"{seed}:split").shuffle(ids)
    n = len(ids)
    n_test = round(n * ratios["test"])
    n_valid = round(n * ratios["valid"])
    return {
        "test": sorted(ids[:n_test]),
        "valid": sorted(ids[n_test : n_test + n_valid]),
        "train": sorted(ids[n_test + n_valid :]),
    }


def _select_measurement_turns(
    turns_by_speaker: dict[str, list[tuple[str, Turn]]],
    split_by_session: dict[str, str],
    cap_sec: float,
) -> dict[str, tuple[list[tuple[str, Turn]], str]]:
    """For each speaker, pick the turns fed to ``measure_speaker``.

    Prefers turns from sessions in the train split; falls back to turns from
    any split when the speaker has none there (a speaker whose only session
    landed in valid/test would otherwise never get measured, and every
    caption needs a voice description). Within the chosen pool, turns are
    taken earliest-first (sorted by session id, then start time) up to a
    cumulative ``cap_sec`` of audio - librosa.pyin's cost scales with input
    duration, and long per-speaker histories don't improve the median F0 /
    F0 IQR estimate enough to justify measuring all of it.

    Returns ``speaker_id -> (selected [(session_id, Turn), ...], source)``
    where ``source`` is ``"train"`` or ``"fallback_any_split"``.
    """
    selected: dict[str, tuple[list[tuple[str, Turn]], str]] = {}
    for speaker_id, entries in turns_by_speaker.items():
        train_entries = [e for e in entries if split_by_session[e[0]] == "train"]
        if train_entries:
            pool, source = train_entries, "train"
        else:
            pool, source = entries, "fallback_any_split"

        pool = sorted(pool, key=lambda e: (e[0], e[1].start))
        picked: list[tuple[str, Turn]] = []
        total = 0.0
        for entry in pool:
            if picked and total >= cap_sec:
                break
            picked.append(entry)
            total += entry[1].end - entry[1].start
        selected[speaker_id] = (picked, source)
    return selected


def _distribution(values: Sequence[float]) -> str:
    if not values:
        return "n=0"
    vals = sorted(values)
    q = statistics.quantiles(vals, n=4) if len(vals) >= 2 else [vals[0]] * 3
    return (
        f"n={len(vals)} min={vals[0]:.1f} p25={q[0]:.1f} median={q[1]:.1f} "
        f"p75={q[2]:.1f} max={vals[-1]:.1f} mean={statistics.fmean(vals):.1f}"
    )


def _write_variant_split(
    out_dir: Path, variant: str, split: str, records: list[dict]
) -> None:
    variant_dir = out_dir / variant / split
    variant_dir.mkdir(parents=True, exist_ok=True)
    dialogues_path = variant_dir / "dialogues.jsonl"
    with dialogues_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    dataset_json = {
        "data_entry": [
            {
                "name": "dialogue",
                "path": str(dialogues_path.resolve()),
                "reader": "dialogue",
            }
        ],
        "samples": sorted(r["example_id"] for r in records),
    }
    (variant_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )


def build(
    dataset_root: str | Path,
    out_dir: str | Path,
    cfg: dict,
    seed: int,
    metadata: dict | None = None,
) -> None:
    """Run the full pipeline: parse manifests, split, window, cut wavs,
    measure speakers, emit both variants, write outputs, print the stats
    block."""
    root = Path(dataset_root)
    out_dir = Path(out_dir).resolve()
    target_sr = int(cfg["target_sample_rate"])

    manifests = root / cfg["manifests_subdir"]
    recordings = load_recordings(
        manifests / "recordings.jsonl.gz", audio_subdir=cfg["audio_subdir"]
    )
    supervisions = load_supervisions(manifests / "supervisions.jsonl.gz", recordings)
    session_ids = sorted(set(recordings) & set(supervisions))

    splits = split_sessions(session_ids, cfg["split_ratios"], seed)
    session_split = {sid: split for split, ids in splits.items() for sid in ids}

    windows_by_split: dict[str, list] = {split: [] for split in SPLITS}
    turns_by_speaker: dict[str, list[tuple[str, Turn]]] = {}
    stats = {split: WindowingStats() for split in SPLITS}

    for sid in session_ids:
        split = session_split[sid]
        turns = merge_turns(supervisions[sid], cfg["merge_gap"])
        for turn in turns:
            turns_by_speaker.setdefault(turn.speaker, []).append((sid, turn))
        records, session_stats = build_windows(
            sid,
            recordings[sid],
            turns,
            window_min=cfg["window_min"],
            window_max=cfg["window_max"],
            boundary_guard=cfg["boundary_guard"],
            tail_min=cfg["tail_min"],
            rng=random.Random(f"{seed}:window:{sid}"),
        )
        stats[split].merge(session_stats)
        windows_by_split[split].extend(records)

    # Gender-metadata shape audit MUST run once, before any measurement.
    all_speaker_ids = sorted(turns_by_speaker)
    audit_gender_metadata(metadata, all_speaker_ids)

    selection = _select_measurement_turns(
        turns_by_speaker, session_split, cfg["measure_cap_sec"]
    )
    attrs_by_speaker = {}
    measure_source = {}
    for speaker_id in all_speaker_ids:
        picked, source = selection[speaker_id]
        turn_wavs = []
        texts = []
        for session_id, turn in picked:
            audio_path = root / recordings[session_id].audio_relpath
            turn_wavs.append(
                load_window_channel(
                    audio_path, turn.start, turn.end, turn.channel, target_sr
                )
            )
            texts.append(turn.text)
        attrs_by_speaker[speaker_id] = measure_speaker(
            turn_wavs, target_sr, texts, speaker_id, metadata=metadata
        )
        measure_source[speaker_id] = source

    # Frozen per-speaker voice description text, kept for the stats block
    # only (see dataset.emit module docstring: this must NOT be threaded
    # back into emit_tac_records/emit_mono_record's `descriptions` override,
    # which expects pre-formatted paraphrase text, not raw template output).
    frozen_descriptions = {
        sid: voice_description(attrs) for sid, attrs in attrs_by_speaker.items()
    }

    tac_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    mono_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    tac_dropped: dict[str, int] = {split: 0 for split in SPLITS}

    for split in SPLITS:
        audio_out_dir = out_dir / "audio" / split
        for window in windows_by_split[split]:
            window_audio = cut_window_wavs(
                window, root, audio_out_dir, target_sr=target_sr
            )
            tac_records = emit_tac_records(window, attrs_by_speaker, window_audio)
            if tac_records:
                tac_by_split[split].extend(tac_records)
            else:
                tac_dropped[split] += 1
            mono_by_split[split].append(
                emit_mono_record(window, attrs_by_speaker, window_audio)
            )

    for split in SPLITS:
        _write_variant_split(out_dir, "tac", split, tac_by_split[split])
        _write_variant_split(out_dir, "mono", split, mono_by_split[split])

    _print_stats(
        seed=seed,
        root=root,
        session_ids=session_ids,
        splits=splits,
        stats=stats,
        windows_by_split=windows_by_split,
        tac_dropped=tac_dropped,
        attrs_by_speaker=attrs_by_speaker,
        measure_source=measure_source,
        tac_by_split=tac_by_split,
        mono_by_split=mono_by_split,
    )


def _print_stats(
    *,
    seed,
    root,
    session_ids,
    splits,
    stats,
    windows_by_split,
    tac_dropped,
    attrs_by_speaker,
    measure_source,
    tac_by_split,
    mono_by_split,
) -> None:
    print(f"BagPiper lm_tts dataset build (seed={seed}, root={root})")
    print(f"  sessions: {len(session_ids)}")
    for split in SPLITS:
        n_windows = len(windows_by_split[split])
        print(
            f"  {split}: {n_windows} windows over {len(splits[split])} sessions; "
            f"tac-dropped {tac_dropped[split]} windows (<2 active speakers); "
            f"tac records {len(tac_by_split[split])}; mono records {len(mono_by_split[split])}"
        )
        st = stats[split]
        print(
            f"    dropped {st.dropped_span_sec:.1f}s oversized blocked spans, "
            f"{st.dropped_sliver_sec:.1f}s slivers, {st.dropped_tail_sec:.1f}s tails, "
            f"{st.dropped_empty_windows} empty windows"
        )
    print("  speakers:")
    for speaker_id in sorted(attrs_by_speaker):
        a = attrs_by_speaker[speaker_id]
        src = measure_source[speaker_id]
        print(
            f"    {speaker_id}: pitch={a.pitch_band} variability={a.variability_band} "
            f"rate={a.rate_band} gender={a.gender} gender_source={a.gender_source} "
            f"measure_source={src}"
        )
    for variant, by_split in (("tac", tac_by_split), ("mono", mono_by_split)):
        lengths = [
            len(r["messages"][1][2].split())
            for records in by_split.values()
            for r in records
        ]
        print(f"  {variant} caption length[words]: {_distribution(lengths)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="SSSD corpus root; falls back to $SSSD_ROOT, then conf/dataset.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory; default <recipe>/data",
    )
    parser.add_argument(
        "--conf",
        default=None,
        help="path to the builder config yaml; default conf/dataset.yaml",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="optional {speaker_id: {gender: ...}} JSON; absent -> pitch heuristic",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    conf_path = (
        Path(args.conf) if args.conf else (_RECIPE_DIR / "conf" / "dataset.yaml")
    )
    cfg = load_config(conf_path)
    root = resolve_dataset_root(args.dataset_root, cfg)
    out_dir = Path(args.out_dir) if args.out_dir else (_RECIPE_DIR / "data")
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    metadata = None
    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text())

    build(root, out_dir, cfg, seed, metadata=metadata)


if __name__ == "__main__":
    main()
